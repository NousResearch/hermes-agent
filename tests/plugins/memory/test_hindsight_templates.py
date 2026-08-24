"""Tests for the Hindsight setup-wizard starter-template step."""

import json
from contextlib import contextmanager

import pytest

from plugins.memory.hindsight import templates as tpl


_CATALOG = {
    "templates": [
        {"id": "conversation", "name": "Conversation", "integrations": ["litellm", "hermes"],
         "manifest_file": "templates/conversation.json"},
        {"id": "coding-agent", "name": "Coding Agent", "integrations": ["claude-code"],
         "manifest_file": "templates/coding-agent.json"},
        {"id": "hermes-gateway-bot", "name": "Gateway Bot", "integrations": ["hermes"],
         "manifest_file": "templates/hermes-gateway-bot.json"},
    ]
}


def test_fetch_hermes_templates_filters_to_hermes(monkeypatch):
    monkeypatch.setattr(tpl, "_get_json", lambda url: _CATALOG)
    entries = tpl.fetch_hermes_templates("https://example/templates.json")
    ids = [e["id"] for e in entries]
    assert ids == ["conversation", "hermes-gateway-bot"]  # coding-agent excluded


def test_fetch_manifest_resolves_relative_url(monkeypatch):
    seen = {}

    def _fake(url):
        seen["url"] = url
        return {"version": "1"}

    monkeypatch.setattr(tpl, "_get_json", _fake)
    tpl.fetch_manifest(
        {"manifest_file": "templates/hermes-gateway-bot.json"},
        "https://raw.example/data/templates.json",
    )
    assert seen["url"] == "https://raw.example/data/templates/hermes-gateway-bot.json"


def test_apply_template_posts_to_import_endpoint(monkeypatch):
    captured = {}

    @contextmanager
    def _fake_open(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["auth"] = req.get_header("Authorization")
        captured["body"] = json.loads(req.data.decode("utf-8"))

        class _Resp:
            def read(self):
                return b""

        yield _Resp()

    monkeypatch.setattr(tpl, "open_credentialed_url", _fake_open)
    tpl.apply_template("https://api.hindsight.vectorize.io/", "hermes", "hsk_abc", {"version": "1"})

    assert captured["url"] == "https://api.hindsight.vectorize.io/v1/default/banks/hermes/import"
    assert captured["method"] == "POST"
    assert captured["auth"] == "Bearer hsk_abc"
    assert captured["body"] == {"version": "1"}


def test_apply_template_omits_auth_when_no_key(monkeypatch):
    captured = {}

    @contextmanager
    def _fake_open(req, timeout=None):
        captured["auth"] = req.get_header("Authorization")

        class _Resp:
            def read(self):
                return b""

        yield _Resp()

    monkeypatch.setattr(tpl, "open_credentialed_url", _fake_open)
    tpl.apply_template("http://localhost:8888", "hermes", None, {"version": "1"})
    assert captured["auth"] is None


def _select_returning(idx):
    def _select(title, items, default=0, cancel_returns=None):
        return idx
    return _select


def _select_seq(*returns):
    it = iter(returns)

    def _select(title, items, default=0, cancel_returns=None):
        return next(it)

    return _select


def test_supported_for_mode():
    assert tpl.supported_for_mode("cloud") is True
    assert tpl.supported_for_mode("local_external") is True
    assert tpl.supported_for_mode("local_embedded") is False
    assert tpl.supported_for_mode("local") is False


def test_run_template_step_applies_selected(monkeypatch):
    monkeypatch.setattr(tpl, "fetch_hermes_templates", lambda url=None: [
        {"id": "hermes-gateway-bot", "name": "Gateway Bot", "manifest_file": "templates/x.json"},
    ])
    monkeypatch.setattr(tpl, "fetch_manifest", lambda entry, url=None: {"version": "1"})
    monkeypatch.setattr(tpl, "probe_existing_customization", lambda *a: False)
    applied = {}
    monkeypatch.setattr(tpl, "apply_template",
                        lambda api_url, bank_id, api_key, manifest: applied.update(bank=bank_id))

    result = tpl.run_template_step(
        api_url="https://api", bank_id="hermes", api_key="k",
        select=_select_returning(0), cancelled=-1, log=lambda *_: None,
    )
    assert result == "hermes-gateway-bot"
    assert applied["bank"] == "hermes"


def test_run_template_step_blank_selection_skips(monkeypatch):
    entries = [{"id": "hermes-gateway-bot", "name": "Gateway Bot", "manifest_file": "templates/x.json"}]
    monkeypatch.setattr(tpl, "fetch_hermes_templates", lambda url=None: entries)
    called = {"applied": False}
    monkeypatch.setattr(tpl, "apply_template",
                        lambda *a, **k: called.update(applied=True))
    # index len(entries) == the "Blank" row
    result = tpl.run_template_step(
        api_url="https://api", bank_id="hermes", api_key="k",
        select=_select_returning(len(entries)), cancelled=-1, log=lambda *_: None,
    )
    assert result is None
    assert called["applied"] is False


def test_run_template_step_no_templates_is_noop(monkeypatch):
    monkeypatch.setattr(tpl, "fetch_hermes_templates", lambda url=None: [])
    result = tpl.run_template_step(
        api_url="https://api", bank_id="hermes", api_key="k",
        select=_select_returning(0), cancelled=-1, log=lambda *_: None,
    )
    assert result is None


def test_run_template_step_swallows_fetch_errors(monkeypatch):
    def _boom(url=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(tpl, "fetch_hermes_templates", _boom)
    # must not raise
    assert tpl.run_template_step(
        api_url="https://api", bank_id="hermes", api_key="k",
        select=_select_returning(0), cancelled=-1, log=lambda *_: None,
    ) is None


def test_run_template_step_swallows_apply_errors(monkeypatch):
    # gap 1: a failed apply (e.g. 401 for an OAuth-only user) must not crash setup.
    import urllib.error

    monkeypatch.setattr(tpl, "fetch_hermes_templates", lambda url=None: [
        {"id": "hermes-gateway-bot", "name": "Gateway Bot", "manifest_file": "templates/x.json"},
    ])
    monkeypatch.setattr(tpl, "fetch_manifest", lambda entry, url=None: {"version": "1"})
    monkeypatch.setattr(tpl, "probe_existing_customization", lambda *a: False)

    def _raise(*a, **k):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(tpl, "apply_template", _raise)
    logs = []
    result = tpl.run_template_step(
        api_url="https://api", bank_id="hermes", api_key=None,
        select=_select_returning(0), cancelled=-1, log=logs.append,
    )
    assert result is None
    assert any("Could not apply" in line for line in logs)


def _fake_open(payload):
    @contextmanager
    def _cm(req, timeout=None):
        class _Resp:
            def read(self):
                return json.dumps(payload).encode("utf-8")

        yield _Resp()

    return _cm


def test_probe_existing_customization_true_when_bank_has_config(monkeypatch):
    monkeypatch.setattr(tpl, "open_credentialed_url",
                        _fake_open({"version": "1", "bank": {"reflect_mission": "x"}}))
    assert tpl.probe_existing_customization("https://api", "hermes", "k") is True


def test_probe_existing_customization_false_when_empty(monkeypatch):
    monkeypatch.setattr(tpl, "open_credentialed_url",
                        _fake_open({"version": "1"}))
    assert tpl.probe_existing_customization("https://api", "hermes", "k") is False


def test_probe_existing_customization_false_on_error(monkeypatch):
    def _boom(req, timeout=None):
        raise OSError("no bank")

    monkeypatch.setattr(tpl, "open_credentialed_url", _boom)
    assert tpl.probe_existing_customization("https://api", "missing", None) is False


def _wire_apply(monkeypatch, customized):
    monkeypatch.setattr(tpl, "fetch_hermes_templates", lambda url=None: [
        {"id": "hermes-gateway-bot", "name": "Gateway Bot", "manifest_file": "templates/x.json"},
    ])
    monkeypatch.setattr(tpl, "fetch_manifest", lambda entry, url=None: {"version": "1"})
    monkeypatch.setattr(tpl, "probe_existing_customization", lambda *a: customized)
    called = {"applied": False}
    monkeypatch.setattr(tpl, "apply_template", lambda *a, **k: called.update(applied=True))
    return called


def test_warns_and_keeps_existing_when_declined(monkeypatch):
    called = _wire_apply(monkeypatch, customized=True)
    # first select = pick template (0); second select = confirm -> "Keep existing" (1)
    result = tpl.run_template_step(
        api_url="https://api", bank_id="hermes", api_key="k",
        select=_select_seq(0, 1), cancelled=-1, log=lambda *_: None,
    )
    assert result is None
    assert called["applied"] is False


def test_warns_then_applies_when_confirmed(monkeypatch):
    called = _wire_apply(monkeypatch, customized=True)
    # pick template (0), confirm "Apply" (0)
    result = tpl.run_template_step(
        api_url="https://api", bank_id="hermes", api_key="k",
        select=_select_seq(0, 0), cancelled=-1, log=lambda *_: None,
    )
    assert result == "hermes-gateway-bot"
    assert called["applied"] is True


def test_fresh_bank_skips_the_warning(monkeypatch):
    called = _wire_apply(monkeypatch, customized=False)
    # only ONE select call (no confirm) — _select_seq with a single value proves it
    result = tpl.run_template_step(
        api_url="https://api", bank_id="hermes", api_key="k",
        select=_select_seq(0), cancelled=-1, log=lambda *_: None,
    )
    assert result == "hermes-gateway-bot"
    assert called["applied"] is True
