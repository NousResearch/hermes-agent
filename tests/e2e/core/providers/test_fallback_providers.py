"""Provider fallback (``fallback_providers``) through REAL ``rabbit -z`` processes.

Two loopback fakes stand in for two vendors: the primary and the fallback. Everything
between the CLI and those sockets is real Rabbit: config loading, credential resolution,
the retry ladder, fallback activation and the fallback client.

Proven here:

* a primary that answers turn 1 and then every request with 503 is tried on turn 2
  (a ``--resume``), then that turn is answered by the configured fallback, which receives
  the FULL conversation (system prompt, turn 1's prompt and answer, turn 2's prompt) and is
  reported as the model that served the turn.
"""

from __future__ import annotations

import sys

import pytest

from tests.e2e.core.providers._openai_helpers import (
    Home,
    bounded_turn,
    chat_messages,
    custom_chat_config,
    db_messages,
)
from tests.fakes.fake_llm_provider import FakeLLMServer
from tests.fakes.providers.chat_variants import CError, CText, FakeChatVariantServer

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="subprocess harness is Linux-gated")

FALLBACK_MODEL = "fallback-model"
PROMPT = "CANARY-PROMPT say hello"
FIRST_PROMPT = "CANARY-FIRST what is two plus two"
# A primary that is never abandoned for the fallback keeps backing off for minutes, so a
# turn overrunning this is a HarnessError (a real failure even under a KNOWN entry).
TURN_BUDGET = 45.0


def _fallback_entry(fallback: FakeLLMServer) -> list[dict]:
    return [{"provider": "custom", "model": FALLBACK_MODEL, "base_url": fallback.base_url}]


def _user_texts(body: dict) -> list[str]:
    return [str(m.get("content")) for m in chat_messages(body, "user")]


def _turns(body: dict) -> list[tuple[str, str]]:
    """``(role, text)`` of every non-system message: the conversation a request carries."""
    return [(m["role"], str(m.get("content") or "")) for m in chat_messages(body) if m.get("role") != "system"]


def test_persistent_primary_503_is_answered_by_fallback_with_full_conversation(tmp_path) -> None:
    def answer_once_then_503(_record: dict) -> CText | CError:
        if not primary.main_records()[:-1]:  # this request is the only one so far: turn 1
            return CText("FIRST-ON-PRIMARY")
        return CError(503, "Service temporarily unavailable", code="service_unavailable")

    with FakeChatVariantServer(answer_once_then_503) as primary, \
            FakeLLMServer(default_text="FROM-FALLBACK") as fallback:
        cfg = custom_chat_config(primary.base_url)
        cfg["fallback_providers"] = _fallback_entry(fallback)
        h = Home(tmp_path).write(cfg, {"OPENAI_API_KEY": "sk-fake"})
        first = bounded_turn(h, FIRST_PROMPT, TURN_BUDGET)
        assert first.proc.returncode == 0 and first.stdout.strip() == "FIRST-ON-PRIMARY", first.describe()
        run = bounded_turn(h, PROMPT, TURN_BUDGET, resume=first.session_id)
        primary_mains = primary.main_requests()
        fallback_mains = fallback.main_requests()

    assert run.proc.returncode == 0 and run.stdout.strip() == "FROM-FALLBACK", run.describe()
    assert len(primary_mains) >= 2, "the primary was never tried on turn 2"
    assert fallback_mains, f"fallback never received a request: {run.describe()}"
    fb = fallback_mains[0]
    assert fb.get("model") == FALLBACK_MODEL, fb.get("model")
    # The fallback gets the whole conversation, not just the prompt that failed over.
    system = chat_messages(fb, "system")
    assert system and str(system[0].get("content") or "").strip() and chat_messages(fb)[0] is system[0], (
        f"fallback request carries no leading system prompt: {chat_messages(fb)[:1]}")
    want = [("user", FIRST_PROMPT), ("assistant", "FIRST-ON-PRIMARY"), ("user", PROMPT)]
    canaries = [(role, next((w for _r, w in want if w in text), None)) for role, text in _turns(fb)]
    assert [c for c in canaries if c[1]] == want, _turns(fb)
    # Exactly what the primary was last asked, minus nothing.
    assert _turns(fb) == _turns(primary_mains[-1]), (_turns(fb), _turns(primary_mains[-1]))
    assert run.usage.get("model") == FALLBACK_MODEL, run.describe()
    answers = [r for r in db_messages(h, run.session_id)
               if r["role"] == "assistant" and "FROM-FALLBACK" in (r.get("content") or "")]
    assert len(answers) == 1, db_messages(h, run.session_id)


