"""Entitlement gate for a model pick that would become the profile default (t_27cf7a7b).

The 2026-09-28 incident: ``custom:zai/glm-5.3-flashx`` (base_url = the Z.AI coding
endpoint) was persisted as ``profiles/hephaestus/config.yaml`` ``model.default``
although the plan refuses it with ``429 code=1311`` — every session afterwards paid
a 429 and then a fallback hop. Nothing on the persist path probed the pick.

These tests pin the gate:
  * only a PROVEN refusal blocks (a throttle, a 5xx, a transport error, a local
    endpoint or a missing credential/endpoint never do);
  * the refusal is printed and the config write does not happen — on both persist
    paths (``_persist_model`` for the setup flows, ``persist_model_selection`` for
    ``/model --global``);
  * ``HERMES_ALLOW_UNENTITLED_PICK=1`` restores warn-and-persist.

All file I/O is against ``tmp_path`` (tests/home_io_guard.py forbids the real home).
"""

from __future__ import annotations

import urllib.error

import pytest

from hermes_cli import model_entitlement_guard as guard

# Live bodies, recorded from the real endpoints on 2026-09-28.
PLAN_REFUSAL_BODY = {"error": {"code": "1311", "message": (
    "Your current subscription plan does not yet include access to GLM-5.3-FlashX")}}
THROTTLE_BODY = {"error": {"message": "Rate limit exceeded. Please retry after 20s.",
                           "type": "rate_limit_error"}}
RETIRED_FREE_BODY = {"error": {"message": (
    "This model is unavailable for free. The paid version is available now - use this "
    "slug instead: upstage/solar-pro4"), "code": 404}}


def _probe_returning(monkeypatch, status: int, body: dict):
    monkeypatch.setattr(guard, "_post_chat_completion",
                        lambda base_url, api_key, model, timeout: (status, body))


def _probe_raising(monkeypatch, exc):
    def boom(base_url, api_key, model, timeout):
        raise exc
    monkeypatch.setattr(guard, "_post_chat_completion", boom)


class TestRefusalProbe:
    def test_entitled_pick_is_allowed(self, monkeypatch):
        _probe_returning(monkeypatch, 200, {"choices": [{"message": {"content": "pong"}}]})
        assert guard.refusal_for_pick(
            "glm-5.3", provider="zai", base_url="https://api.z.ai/api/coding/paas/v4",
            api_key="k") is None

    def test_plan_refusal_is_reported(self, monkeypatch):
        _probe_returning(monkeypatch, 429, PLAN_REFUSAL_BODY)
        reason = guard.refusal_for_pick(
            "glm-5.3-flashx", provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4", api_key="k")
        assert reason and "include access" in reason

    def test_retired_free_slug_is_reported(self, monkeypatch):
        _probe_returning(monkeypatch, 404, RETIRED_FREE_BODY)
        assert guard.refusal_for_pick(
            "upstage/solar-pro4:free", provider="openrouter",
            base_url="https://openrouter.ai/api/v1", api_key="k")

    def test_throttle_is_not_a_refusal(self, monkeypatch):
        """A 429 window means "busy", not "not entitled": never block on it."""
        _probe_returning(monkeypatch, 429, THROTTLE_BODY)
        assert guard.refusal_for_pick(
            "glm-5.3", provider="zai", base_url="https://api.z.ai/api/coding/paas/v4",
            api_key="k") is None

    def test_server_error_is_not_a_refusal(self, monkeypatch):
        _probe_returning(monkeypatch, 503, {"error": {"message": "Service temporarily overloaded"}})
        assert guard.refusal_for_pick(
            "glm-5.3", provider="zai", base_url="https://api.z.ai/api/coding/paas/v4",
            api_key="k") is None

    def test_transport_error_is_not_a_refusal(self, monkeypatch):
        """A flaky network must never block a pick."""
        _probe_raising(monkeypatch, urllib.error.URLError("no route to host"))
        assert guard.refusal_for_pick(
            "glm-5.3", provider="zai", base_url="https://api.z.ai/api/coding/paas/v4",
            api_key="k") is None

    def test_local_endpoint_is_not_probed(self, monkeypatch):
        def forbidden(*a, **kw):  # pragma: no cover - must not be reached
            raise AssertionError("a local endpoint must not be probed")
        monkeypatch.setattr(guard, "_post_chat_completion", forbidden)
        assert guard.refusal_for_pick(
            "qwen2.5vl:7b", provider="ollama-launch",
            base_url="http://127.0.0.1:11434/v1", api_key="k") is None

    def test_missing_endpoint_or_key_is_conclusive_nothing(self, monkeypatch):
        def forbidden(*a, **kw):  # pragma: no cover - must not be reached
            raise AssertionError("no endpoint/key means nothing to probe")
        monkeypatch.setattr(guard, "_post_chat_completion", forbidden)
        assert guard.refusal_for_pick("m", provider="p", base_url="", api_key="k") is None
        assert guard.refusal_for_pick("m", provider="p", base_url="https://x/v1", api_key="") is None


class TestPolicy:
    def test_default_policy_refuses_and_names_the_override(self, monkeypatch, capsys):
        _probe_returning(monkeypatch, 429, PLAN_REFUSAL_BODY)
        monkeypatch.delenv(guard.POLICY_ENV, raising=False)
        allowed = guard.ensure_pick_entitled(
            "glm-5.3-flashx", provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4", api_key="k")
        out = capsys.readouterr().out
        assert allowed is False
        assert "was NOT saved as the default" in out
        assert "hermes -m glm-5.3-flashx" in out
        assert guard.POLICY_ENV in out

    def test_escape_hatch_warns_and_persists(self, monkeypatch, capsys):
        _probe_returning(monkeypatch, 429, PLAN_REFUSAL_BODY)
        monkeypatch.setenv(guard.POLICY_ENV, "1")
        allowed = guard.ensure_pick_entitled(
            "glm-5.3-flashx", provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4", api_key="k")
        out = capsys.readouterr().out
        assert allowed is True
        assert "Saving it anyway" in out

    def test_refusal_copy_never_echoes_a_credential(self, monkeypatch, capsys):
        _probe_returning(monkeypatch, 429, PLAN_REFUSAL_BODY)
        monkeypatch.delenv(guard.POLICY_ENV, raising=False)
        guard.ensure_pick_entitled(
            "glm-5.3-flashx", provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4", api_key="sk-super-secret")
        assert "sk-super-secret" not in capsys.readouterr().out


@pytest.fixture
def seeded_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a previous default to fall back to.

    ``HERMES_HOME`` is set BEFORE any Hermes module resolves a path, and
    ``tests/home_io_guard.py`` refuses any I/O that escapes to the real home —
    a stray write here fails loudly instead of touching the live config.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: deepseek-v4.1-flash\n  provider: cheaper-inference\n"
        "  base_url: https://api.cheaperinference.com/v1\n", encoding="utf-8")
    return tmp_path


def _default(home) -> str:
    import hermes_yaml as yaml
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["model"]["default"]


class TestSetupFlowPersistGate:
    def test_refused_pick_leaves_the_default_untouched(self, seeded_home, monkeypatch, capsys):
        from hermes_cli.model_setup_flows_common import _finish_model
        monkeypatch.setattr(guard, "ensure_pick_entitled", lambda *a, **kw: False)
        result = _finish_model(
            "glm-5.3-flashx", "custom:zai",
            "Default model set to: glm-5.3-flashx",
            base_url="https://api.z.ai/api/coding/paas/v4")
        out = capsys.readouterr().out
        assert result is None
        assert "deepseek-v4.1-flash" == _default(seeded_home)
        assert "Default model set to" not in out
        assert "No change." in out

    def test_allowed_pick_is_persisted(self, seeded_home, monkeypatch):
        from hermes_cli.model_setup_flows_common import _finish_model
        monkeypatch.setattr(guard, "ensure_pick_entitled", lambda *a, **kw: True)
        _finish_model(
            "glm-5.3", "zai", "Default model set to: glm-5.3",
            base_url="https://api.z.ai/api/coding/paas/v4")
        assert "glm-5.3" == _default(seeded_home)


class TestModelSwitchPersistGate:
    def _run(self, home, monkeypatch, allowed: bool):
        import hermes_yaml as yaml
        from hermes_cli.model_switch import ModelSwitchResult, persist_model_selection
        monkeypatch.setattr(guard, "ensure_pick_entitled", lambda *a, **kw: allowed)
        result = ModelSwitchResult(
            success=True, new_model="glm-5.3-flashx", target_provider="custom:zai",
            provider_changed=True, api_key="k",
            base_url="https://api.z.ai/api/coding/paas/v4", api_mode="", is_global=True)
        persist_model_selection(result, config_path=home / "config.yaml")
        return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["model"]

    def test_refused_global_persist_keeps_the_previous_default(self, seeded_home, monkeypatch, capsys):
        model = self._run(seeded_home, monkeypatch, allowed=False)
        assert model["default"] == "deepseek-v4.1-flash"
        assert "was NOT changed" in capsys.readouterr().out

    def test_allowed_global_persist_writes_the_pick(self, seeded_home, monkeypatch):
        model = self._run(seeded_home, monkeypatch, allowed=True)
        assert model["default"] == "glm-5.3-flashx"
