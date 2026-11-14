"""The Anthropic 401 help names a callable key's credential source, not always Entra ID.

#154707: a provider backed by ``key_cmd`` (``CommandTokenSource``) is callable, so the 401 help
took the Entra ID branch and told the user to run ``az login`` — while the key banner for the same
key already said ``key_cmd token (command)``. The branch must read the source tag, never "is it
callable", and no display path may invoke the callable.
"""

from __future__ import annotations

from agent import credential_source as cs
from agent.i18n import t


def _capture_401(monkeypatch, key) -> str:
    import agent.turn_recovery as tr

    printed = []
    monkeypatch.setattr(tr, "_plines", lambda agent, *lines: printed.extend(lines))
    tr._print_anthropic_401_diagnostics(object(), key)
    return "\n".join(printed)


def _never_called_provider():
    def provider() -> str:
        raise AssertionError("the 401 help invoked the credential callable")

    return provider


def test_key_cmd_401_help_names_its_source_not_entra(monkeypatch):
    key = cs.tag_credential_source(_never_called_provider(), cs.KEY_CMD)

    out = _capture_401(monkeypatch, key)

    assert "Microsoft Entra ID" not in out, out
    assert "az login" not in out, out
    assert t(cs.SOURCE_LABEL_KEYS[cs.KEY_CMD]) in out, out


def test_entra_id_401_help_still_reports_entra(monkeypatch):
    key = cs.tag_credential_source(_never_called_provider(), cs.ENTRA_ID)

    out = _capture_401(monkeypatch, key)

    assert "Microsoft Entra ID" in out, out
    assert "az login" in out, out
