# Forward-compat: applying these 14 files onto v0.17.0

This PR carries 14 overlay files whose **diff is computed against v0.16.0**
(`3c231eb3979ab9c57d5cd6d02f1d577a3b718b43`), the base the whole campaign is cut
from. Between v0.16.0 and **v0.17.0** (`2bd1977d8fad185c9b4be47884f7e87f1add0ce3`)
upstream landed drift in these same 14 files, so a naive 3-way apply of the
v0.16.0-based diff onto v0.17.0 produces conflicts (29 conflict hunks total).

That is expected and benign: it is the normal cost of rebasing a feature branch
across a release. To make the PR **directly pullable onto v0.17.0** without the
operator re-deriving the merge, the conflict-free, v0.17.0-ready variant of every
one of the 14 files is committed under [`v0.17.0-ready/`](./v0.17.0-ready/).

## What `v0.17.0-ready/` is

Each file there = **upstream v0.17.0 baseline + this overlay's added behavior**,
with every conflict resolved and **zero conflict markers**. The resolution
classified each of the 29 conflict hunks as complementary-additive (keep both),
refactor-of-our-code (adopt upstream structure, preserve our behavior), or
ours-only/theirs-only, and merged accordingly. Overlay-unique behavior was
verified to survive (autopilot hooks, prelude tier, payment-fallback loop, MCP
in-flight guard, trigram tokenizer, reasoning-callback threading, etc.).

## Verification (reproducible)

```
# fresh v0.17.0 tree
git worktree add --detach /tmp/v017 2bd1977d8fad185c9b4be47884f7e87f1add0ce3
cd /tmp/v017
# drop in the 14 ready files
for f in $(cat <PR>/v0.17.0-ready/MANIFEST.txt); do
  cp <PR>/v0.17.0-ready/"$f" "$f"
done
# result: 0 conflict markers, every changed file compiles
grep -rlE '^<<<<<<<|^>>>>>>>' --include='*.py' . | wc -l     # -> 0
for f in $(git diff --name-only); do python3 -m py_compile "$f"; done   # -> all pass
```

Combined with the sibling clean-residual PR (which applies on v0.17.0 with no
conflicts), the full residual set lands on pristine v0.17.0 as **34 files
changed, +6050/-494, 0 conflict markers, 0 compile failures.**

## Per-file conflict count (resolved in `v0.17.0-ready/`)

| File | conflict hunks | resolution shape |
|---|---|---|
| agent/anthropic_adapter.py | 2 | refactor + complementary-additive |
| agent/auxiliary_client.py | 2 | semantic-overlap (kept overlay recovery loop) |
| agent/conversation_loop.py | 2 | ours-only block + TurnRetryState refactor rewire |
| agent/system_prompt.py | 2 | complementary-additive + prelude-aware join |
| cli.py | 2 | ours-only (`_init_agent`, `_toggle_autopilot`) |
| gateway/platforms/api_server.py | 1 | semantic-overlap (session-ctx + reasoning_callback) |
| gateway/run.py | 1 | spliced full overlay handler region |
| hermes_cli/main.py | 3 | complementary-additive + dedup truncated fragments |
| hermes_state.py | 4 | complementary-additive + connect-path refactor |
| tests/agent/test_auxiliary_client.py | 1 | complementary-additive (kept both tests) |
| tests/hermes_cli/test_inventory.py | 1 | complementary-additive (kept all 6 tests) |
| tools/mcp_tool.py | 4 | complementary-additive + combine |
| tools/skills_tool.py | 2 | refactor-of-our-code (adopt upstream form) |
| tui_gateway/server.py | 2 | complementary-additive |
