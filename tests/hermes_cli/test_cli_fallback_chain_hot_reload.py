"""An open classic-CLI chat adopts a ``fallback_providers`` chain added after it started (#95066).

``HermesCLI`` holds one long-lived agent and read the chain once in ``__init__``; ``hermes fallback
add`` from another terminal never reached the open chat. The turn loop (``HermesCLI.chat``) now
re-reads the chain fail-closed, so a torn config.yaml keeps the last known-good chain.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli
from hermes_cli.config import get_config_path

FALLBACK = [{"provider": "xai-oauth", "model": "grok-4.6"}]


class _StopAfterSync(Exception):
    pass


def _chat_turn(monkeypatch, shell, config_text: str) -> None:
    """Drive ``HermesCLI.chat`` past the fallback sync against ``config_text`` as the live config.yaml."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(shell, "_ensure_runtime_credentials", lambda: True)
    monkeypatch.setattr(shell, "_resolve_turn_agent_config",
                        lambda message: {"signature": shell._active_agent_route_signature, "model": None, "runtime": None})
    monkeypatch.setattr(shell, "_init_agent", lambda **kw: True)

    def stop(message, images):
        raise _StopAfterSync()

    monkeypatch.setattr(shell, "_chat_route_images", stop)
    with pytest.raises(_StopAfterSync):
        shell.chat("hello")


def test_chat_turn_adopts_chain_added_after_the_cli_opened_and_keeps_it_on_torn_config(monkeypatch):
    shell = cli.HermesCLI(compact=True, max_turns=1)
    shell.agent = SimpleNamespace(
        _fallback_chain=[], _fallback_model=None, _fallback_index=0,
        _fallback_activated=False, _rate_limited_until=0, _unavailable_fallback_keys=set(),
    )

    _chat_turn(monkeypatch, shell, "fallback_providers:\n  - provider: xai-oauth\n    model: grok-4.6\n")
    assert shell.agent._fallback_chain == FALLBACK
    assert shell.agent._fallback_model == FALLBACK[0]
    assert shell._fallback_model == FALLBACK  # a rebuilt agent starts from the fresh chain too

    # Torn mid-edit write: keep the last known-good chain rather than wiping it.
    _chat_turn(monkeypatch, shell, "fallback_providers: [\n  - provider: {{{\n")
    assert shell.agent._fallback_chain == FALLBACK
