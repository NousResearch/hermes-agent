"""Live end-to-end proof that the background-review fork ACTUALLY writes mem0.

This is the productionized version of the manual harness Apollo used on
2026-06-30 to prove the bgr→mem0 path after PR #116/#118/#127. It drives the
*real* production path:

    AIAgent.run_conversation()
        → turn_finalizer fires the memory nudge (forced to fire this turn)
        → _spawn_background_review() daemon thread
        → _run_review_in_thread() forks a review AIAgent (skip_memory=True)
        → the fork calls mem0_remember
        → a row lands in the live mem0 store with write_origin=background_review

It is NOT a direct mem0_remember call and NOT a mock — it exercises the same
fork the gateway spawns every ~10 user turns.

GATED OFF BY DEFAULT. It needs the live mem0 store on ACE-AI and a reachable
relay, so it only runs when BGR_MEM0_LIVE_E2E=1. In CI / the normal suite it
skips. Run it manually:

    BGR_MEM0_LIVE_E2E=1 \
    BGR_MEM0_E2E_SSH=ace@192.168.1.216 \
    BGR_MEM0_E2E_RELAY=http://localhost:18811/v1 \
    BGR_MEM0_E2E_MODEL=claude-opus-4-8 \
        venv/bin/python -m pytest tests/run_agent/test_background_review_mem0_e2e.py -s

Three confounds this test encodes (each produced a false "0 rows" before it
worked — see mem0-selfhost-ops skill):

1. The fork does ``contextlib.redirect_stdout(devnull)`` PROCESS-WIDE, so any
   print() from the main thread vanishes mid-run. We capture the fork's tool
   calls by monkeypatching ``summarize_background_review_actions`` (handed the
   fork's completed review_messages), not via stdout.
2. A standalone process does NOT translate OpenAI↔Anthropic tool schemas the
   way the gateway does, so we drive the OpenAI-native relay and disable
   streaming (the bpp relay is fail-closed on streaming).
3. The test FACT must be a *natural* durable preference. A self-referential
   "prove this premise" fact is correctly refused as injection-shaped; and if
   the parent still has mem0_conclude it saves the fact in the foreground,
   leaving the fork nothing new. So we strip mem0_conclude from the parent and
   use a plain preference, making the fork's mem0_remember the only writer.
"""

import os
import re
import subprocess
import threading
import time
import uuid

import pytest

LIVE = os.getenv("BGR_MEM0_LIVE_E2E") == "1"

# The pytest suite auto-redirects HERMES_HOME to a temp dir (AGENTS.md). A LIVE
# e2e must read the REAL config (for the feature flag) and hit the REAL store, so
# pin HERMES_HOME back to the real home when running live. Default ~/.hermes;
# override with BGR_MEM0_E2E_HOME.
if LIVE:
    os.environ["HERMES_HOME"] = os.getenv(
        "BGR_MEM0_E2E_HOME", os.path.expanduser("~/.hermes")
    )

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="live bgr→mem0 e2e; set BGR_MEM0_LIVE_E2E=1 (needs live mem0 store + relay)",
)

SSH = os.getenv("BGR_MEM0_E2E_SSH", "ace@192.168.1.216")
RELAY = os.getenv("BGR_MEM0_E2E_RELAY", "http://localhost:18811/v1")
MODEL = os.getenv("BGR_MEM0_E2E_MODEL", "claude-opus-4-8")
PG_CONTAINER = os.getenv("BGR_MEM0_E2E_PG", "mem0-selfhost-postgres-1")


def _psql(sql: str) -> str:
    proc = subprocess.run(
        [
            "ssh", SSH, "docker", "exec", "-i", PG_CONTAINER,
            # -tA's default field sep is already '|'; do NOT pass '-F |' through
            # ssh (ssh joins remote argv via a shell → the pipe becomes a pipeline).
            "psql", "-U", "mem0_pg", "-d", "mem0", "-tA",
        ],
        input=sql, text=True, capture_output=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return proc.stdout.strip()


def _bgr_count() -> int:
    out = _psql(
        "select count(*) from memories "
        "where payload->>'write_origin'='background_review';"
    )
    return int(out or "0")


def test_background_review_fork_writes_mem0_live():
    # A conftest autouse fixture redirects HERMES_HOME to a temp dir per-test;
    # re-pin it to the real home now (after fixtures have run) so config + the
    # mem0 plugin client resolve against the live deployment.
    os.environ["HERMES_HOME"] = os.getenv(
        "BGR_MEM0_E2E_HOME", os.path.expanduser("~/.hermes")
    )

    from run_agent import AIAgent
    from hermes_cli.config import load_config_readonly, cfg_get
    import agent.background_review as bgr

    # Precondition: the feature flag must be on, else the fork is denied the tool.
    flag = bool(cfg_get(load_config_readonly(), "memory",
                        "background_review_mem0_write", default=False))
    assert flag, "memory.background_review_mem0_write must be true for this e2e"

    token = f"BGR_E2E_{uuid.uuid4().hex[:12]}"
    # Natural durable preference (NOT self-referential — that gets refused as
    # injection-shaped). Token rides in a parenthetical for cleanup.
    fact = (
        "By the way, for future reference: I take my coffee as a flat white with "
        "oat milk, and I never want meetings booked before 11am. "
        f"(ref {token})"
    )

    agent = AIAgent(
        quiet_mode=True,
        max_iterations=6,
        model=MODEL,
        provider="openai",
        base_url=RELAY,
        api_key="x",
        api_mode="chat_completions",
    )
    agent._disable_streaming = True  # relay is fail-closed on streaming

    # Preconditions on the parent.
    assert "memory" in agent.valid_tool_names
    assert "mem0_remember" in agent.valid_tool_names, "PR #118 not active (tool not resident)"
    assert getattr(agent, "_memory_store", None), "parent has no memory store"

    # Strip mem0_conclude from the PARENT so the foreground turn cannot save the
    # fact itself; the fork builds its OWN whitelist (memory, skills,
    # +mem0_remember) and is unaffected.
    try:
        agent.valid_tool_names.discard("mem0_conclude")
    except AttributeError:
        agent.valid_tool_names = [t for t in agent.valid_tool_names if t != "mem0_conclude"]

    # Force the normal memory nudge to fire on THIS turn (no config edit).
    agent._turns_since_memory = max(0, int(agent._memory_nudge_interval) - 1)

    # Capture what the fork actually did (stdout is hijacked, so spy the summary).
    fork_tool_calls: list[str] = []
    _orig = bgr.summarize_background_review_actions

    def _spy(review_messages, prior_snapshot, notification_mode="on"):
        for m in review_messages:
            if m.get("role") == "assistant":
                for tc in (m.get("tool_calls") or []):
                    fork_tool_calls.append((tc.get("function") or {}).get("name", ""))
        return _orig(review_messages, prior_snapshot, notification_mode=notification_mode)

    bgr.summarize_background_review_actions = _spy

    baseline = _bgr_count()
    # Timestamp boundary for cleanup: the model REWORDS facts, so text-matching is
    # fragile (it can miss a reworded second fact). Delete by created_at window in
    # the finally block instead. Use the store's own clock to avoid skew.
    # (_psql already .strip()s its output.)
    started_at = _psql("select now() at time zone 'utc';")
    # Defensive: started_at is composed into SQL below via f-string. It's PG's own
    # now() output (no injection vector), but assert the expected timestamp shape
    # so a future change that derives it from less-trusted input fails loudly here.
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", started_at), started_at
    try:
        result = agent.run_conversation(fact)
        assert result.get("final_response") is not None

        # Wait for the bg-review daemon thread to spawn AND finish. Thread.join()
        # with a timeout returns whether the thread finished OR the timeout
        # expired, so assert is_alive() is False afterward — a still-running fork
        # means the review didn't actually complete in budget.
        spawned = False
        finished = False
        deadline = time.time() + 180
        while time.time() < deadline:
            threads = [t for t in threading.enumerate() if t.name == "bg-review"]
            if threads:
                spawned = True
                for t in threads:
                    t.join(timeout=max(1, deadline - time.time()))
                finished = all(not t.is_alive() for t in threads)
                break
            time.sleep(1)
        assert spawned, "bg-review thread never spawned (memory review did not fire)"
        assert finished, "bg-review thread did not finish within budget"

        # The fork must have invoked the mem0 writer.
        assert "mem0_remember" in fork_tool_calls, (
            f"fork did not call mem0_remember; tool calls were {fork_tool_calls}"
        )

        # And a real row must have landed (async write — poll briefly). Match on
        # write_origin + the post-start window, not on (reworded) fact text.
        landed_id = None
        for _ in range(24):
            rows = _psql(
                "select id from memories "
                "where payload->>'write_origin'='background_review' "
                f"and (payload->>'created_at') > '{started_at}' limit 1;"
            )
            if rows:
                landed_id = rows.splitlines()[0].strip()
                break
            time.sleep(5)
        assert landed_id, "no background_review row landed in the live store"
        assert _bgr_count() > baseline
    finally:
        bgr.summarize_background_review_actions = _orig
        # Clean up by created_at window (the model rewords facts, so text-match
        # is unreliable). Remove only background_review rows this run created.
        try:
            _psql(
                "delete from memories "
                "where payload->>'write_origin'='background_review' "
                f"and (payload->>'created_at') > '{started_at}';"
            )
        except Exception:
            pass
