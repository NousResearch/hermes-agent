"""A callable API key is labelled by its credential source, never by "is it callable".

``/config`` (``cli.py`` ``show_config``) and the agent's key banner (``_print_key_banner``) printed
"Microsoft Entra ID" for every callable key, so a custom provider's ``key_cmd`` bearer or MiniMax
OAuth read as Azure. Each builder now tags its callable with a source, and both surfaces resolve the
label through one table in ``agent.credential_source``. An untagged callable gets a neutral label.
No test may let the display path call the callable: that would mint a token just to mask it.
"""

from __future__ import annotations

import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

from agent.i18n import t


def _never_called() -> str:
    raise AssertionError("a display path invoked the credential callable")


def _config_api_line(capsys, api_key) -> str:
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.api_key, cli.agent = api_key, None
    cli.model, cli.base_url = "m", "https://example.invalid/v1"
    cli.max_turns, cli.enabled_toolsets, cli.verbose = 10, [], False
    cli.session_start = datetime(2031, 1, 1)
    cli.show_config()
    out = capsys.readouterr().out
    return next(line for line in out.splitlines() if t("cli.config.label_api_key") in line)


def _banner(capsys, key) -> str:
    from agent.agent_init import _print_key_banner

    _print_key_banner(key, "API key")
    return capsys.readouterr().out


def _entra_provider(monkeypatch):
    from agent import azure_identity_adapter as adapter

    fake = SimpleNamespace(get_bearer_token_provider=lambda credential, scope: (lambda: _never_called()))
    monkeypatch.setitem(sys.modules, "azure", SimpleNamespace(identity=fake))
    monkeypatch.setitem(sys.modules, "azure.identity", fake)
    monkeypatch.setattr(adapter, "_require_azure_identity", lambda: fake)
    monkeypatch.setattr(adapter, "build_credential", lambda config: object())
    return adapter.build_token_provider()


def _vertex_provider(monkeypatch):
    import agent.vertex_adapter as va

    monkeypatch.setattr(va, "get_vertex_credentials", lambda *_a, **_k: _never_called())
    return va.vertex_access_token


def _key_cmd_provider(monkeypatch):
    import agent.command_token_source as cts

    monkeypatch.setattr(cts, "_mint", lambda *_a, **_k: _never_called())
    return cts.build_command_token_provider("print-token", "corp")


def _minimax_provider(monkeypatch):
    import hermes_cli.auth_minimax as mm

    monkeypatch.setattr(mm, "_minimax_fresh_state", lambda: _never_called())
    return mm.build_minimax_oauth_token_provider()


def _untagged_provider(monkeypatch):
    def some_plugin_token_provider() -> str:
        return _never_called()

    return some_plugin_token_provider


CASES = [
    pytest.param(_entra_provider, "cli.config.microsoft_entra_id", id="entra"),
    pytest.param(_vertex_provider, "cli.config.google_vertex_oauth", id="vertex"),
    pytest.param(_key_cmd_provider, "cli.config.key_cmd_token", id="key_cmd"),
    pytest.param(_minimax_provider, "cli.config.minimax_oauth", id="minimax"),
    pytest.param(_untagged_provider, "cli.config.token_provider", id="unknown"),
]


@pytest.mark.parametrize(("build", "label_key"), CASES)
def test_config_labels_a_callable_key_by_its_source(capsys, monkeypatch, build, label_key):
    line = _config_api_line(capsys, build(monkeypatch))
    assert t(label_key) in line, line
    if label_key != "cli.config.microsoft_entra_id":
        assert t("cli.config.microsoft_entra_id") not in line, line


@pytest.mark.parametrize(("build", "label_key"), CASES)
def test_key_banner_labels_a_callable_key_by_its_source(capsys, monkeypatch, build, label_key):
    out = _banner(capsys, build(monkeypatch))
    assert t(label_key) in out, out
    if label_key != "cli.config.microsoft_entra_id":
        assert "Microsoft Entra ID" not in out, out


def test_both_surfaces_share_one_source_table():
    from agent import credential_source as cs

    labels = {cs.credential_label(cs.tag_credential_source(lambda: "", s)) for s in cs.SOURCE_LABEL_KEYS}
    assert labels == {t(k) for k in cs.SOURCE_LABEL_KEYS.values()}
    assert t(cs.UNKNOWN_LABEL_KEY) not in labels, "the neutral label must not collide with a vendor label"


def test_string_keys_keep_the_masked_form(capsys):
    assert _banner(capsys, "sk-live-abcdefghijklmnop3456").strip() == "🔑 Using API key: sk-live-...3456"
