"""fallback.confirm: ask before leaving the primary; answer cached per turn."""
import types
from unittest.mock import patch

from agent.fallback_cooldown import fallback_consent_blocks


def _agent(answer):
    asked = []
    def cb(q, choices):
        asked.append(choices)
        return answer(choices)
    a = types.SimpleNamespace(clarify_callback=cb, _fallback_activated=False, _fallback_index=0,
                              _fallback_chain=[{"provider": "openrouter", "model": "m"}], model="primary")
    return a, asked


def _cfg(on):
    return patch("hermes_cli.config.load_config", return_value={"fallback": {"confirm": on}})


def test_go_allows_switch():
    a, _ = _agent(lambda c: c[0])
    with _cfg(True):
        assert fallback_consent_blocks(a) is False


def test_stop_blocks_and_is_cached():
    a, asked = _agent(lambda c: c[1])
    with _cfg(True):
        assert fallback_consent_blocks(a) and fallback_consent_blocks(a)
    assert len(asked) == 1


def test_off_headless_and_timeout():
    a, asked = _agent(lambda c: c[0])
    with _cfg(False):
        assert fallback_consent_blocks(a) is False
    assert not asked
    a.clarify_callback = None
    with _cfg(True):
        assert fallback_consent_blocks(a) is False
    b, _ = _agent(lambda c: "")
    with _cfg(True):
        assert fallback_consent_blocks(b) is True
