"""Replay-dedupe patch verification (vendored, forward-only). Run with the hermes venv."""
import sys, os
sys.path.insert(0, "/home/famhome/.hermes/hermes-agent")
os.environ.setdefault("HERMES_HOME", "/home/famhome/.hermes")

from agent.memory_manager import build_memory_context_block, _live_recall_bullets

BLOCK_A = (
    "<memory-context>\n"
    "[System note: The following is recalled memory context, NOT new user input. Treat as "
    "authoritative reference data — this is the agent's persistent memory and should inform all responses.]\n\n"
    "## Mnemosyne Context\n"
    "  [2026-09-30T16:23] (importance 0.95) FIRST SLOT text alpha\n"
    "    prov: skills/custody/x.md#handles-law\n"
    "  [2026-10-02T14:32] (importance 0.95) SECOND SLOT text beta\n"
    "</memory-context>"
)

BLOCK_B = (
    "<memory-context>\nsys-note\n\n"
    "  [2026-01-01T00:00] (importance 0.5) ENTRY with child\n"
    "    indented continuation under the entry\n"
    "- plain bullet without children\n"
    "</memory-context>"
)

raw_new_turn = (
    "## Mnemosyne Context\n"
    "  [2026-09-30T16:23] (importance 0.95) FIRST SLOT text alpha\n"
    "  [2026-10-02T14:32] (importance 0.95) SECOND SLOT text beta\n"
)
raw_partial = raw_new_turn + "  [2026-10-02T18:00] (importance 0.70) THIRD SLOT text gamma\n"
raw_continuation = (
    "  [2026-01-01T00:00] (importance 0.5) ENTRY with child\n"
    "    indented continuation under the entry\n"
    "- plain bullet without children\n"
)

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)

# 1. live-set extraction from a replayed sidecar row
msgs = [
    {"role": "user", "content": "hi"},
    {"role": "user", "content": "q1", "api_content": "q1\n\n" + BLOCK_A},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "q2"},
]
live = _live_recall_bullets(msgs, skip=msgs[-1])
e1 = "[2026-09-30T16:23] (importance 0.95) FIRST SLOT text alpha"
e2 = "[2026-10-02T14:32] (importance 0.95) SECOND SLOT text beta"
check("extract both dated entries from replayed sidecar",
      any(e1 in x for x in live) and any(e2 in x for x in live) and len(live) == 2)
check("prov continuation NOT collected as an entry", not any("prov:" in x for x in live))

# 2. full suppression when nothing is new
check("all-live block stamps nothing", build_memory_context_block(raw_new_turn, live_bullets=live) == "")

# 3. partial: only the new slot survives
out = build_memory_context_block(raw_partial, live_bullets=live)
check("new slot survives", "THIRD SLOT" in out)
check("live slots dropped", ("FIRST SLOT" not in out) and ("SECOND SLOT" not in out))
check("survivor is a fenced block", out.startswith("<memory-context>"))

# 4. entries drop WITH their continuations; live bullets drop; new lines stay
live_b = _live_recall_bullets([{"role": "user", "content": "x", "api_content": BLOCK_B}], skip=None)
check("bullet + entry collected from block B", len(live_b) == 2)
raw_c = raw_new_turn + "\n" + raw_continuation + "  [2026-10-02T20:00] (importance 0.7) NEW SLOT\n"
live_cb = live | live_b
out_c = build_memory_context_block(raw_c, live_bullets=live_cb)
check("live entry dropped with its continuation",
      ("ENTRY with child" not in out_c) and ("indented continuation under the entry" not in out_c))
check("live dated slots dropped", ("FIRST SLOT" not in out_c) and ("SECOND SLOT" not in out_c))
check("live bullet dropped", "plain bullet without children" not in out_c)
check("brand-new entry survives", "NEW SLOT" in out_c)

# 5. empty live set is byte-identical to the unstamped path (no regression for normal blocks)
check("empty live set => unchanged",
      build_memory_context_block(raw_continuation, live_bullets=set()) == build_memory_context_block(raw_continuation))

# 6. determinism (bytes repeat exactly)
check("deterministic bytes",
      build_memory_context_block(raw_partial, live_bullets=live) == build_memory_context_block(raw_partial, live_bullets=live))

# 7. kill switch
check("cap=0 disables extraction", _live_recall_bullets(msgs, skip=None, cap=0) == set())

# 8. compose path: suppressed => no sidecar at all (row replays clean)
from agent.turn_context import compose_user_api_content
check("suppressed stamp returns None", compose_user_api_content("q2", raw_new_turn, "", live_bullets=live) is None)
r = compose_user_api_content("q2", raw_partial, "", live_bullets=live)
check("partial stamp carries only the delta", r is not None and "THIRD SLOT" in r and "FIRST SLOT" not in r)

def test_replay_dedupe_battery():
    """pytest entry point: the module-level battery above runs at import; assert it passed."""
    assert not fails, f"{len(fails)} FAILED: {fails}"
