# schema_dedupe — live A/B for tool-schema text changes on memory, manage_connections, search_files, terminal

Validates a schema-text-only change (each rule stated once instead of in both the tool description and a
parameter description) on the tools where losing a rule would cost the most. Real in-process `AIAgent`s
from two plain checkouts (base = main, pr = the branch), one subprocess per cell with a temp `HERMES_HOME`,
host side effects sandboxed by `../core_tool_deferral/sandbox.py`.

## Tasks

| id | what it proves |
|---|---|
| `mem_swap_stale` | nearly full store (2,138 of 2,200 chars): a stale entry is replaced in one call, durable entries kept |
| `mem_full_add` | nearly full store, user names what is obsolete: remove + add arrive as one batch, not repeated |
| `mem_add_overflow` | nearly full store, user names nothing: the add still lands (batched up front, or one batch after the rejection) |
| `mem_targets` | a fact about the user goes to `user`, an environment fact to `memory` |
| `mem_procedure_to_skill` | a reusable procedure goes to a skill, not into memory |
| `conn_hosted_and_mcp` | hosted target shape (`connect`, bare slug) and MCP target shape (`install`, `mcp: true`) in one request |
| `conn_ambiguous_hosted` | a name on both sides with no MCP wording resolves to hosted |
| `conn_mcp_words` | the same name with "MCP server" wording resolves to MCP |
| `conn_second_account` | a second account is `connect` with a new `alias` |
| `conn_switch_account` | an account switch is `reconnect` with `force: true` |
| `conn_skip_no_reask` | after the user skips, the model does not ask again on its own |
| `conn_no_disconnect` | a disconnect request is declined and pointed at the Portal / desktop app |
| `search_content`, `search_newest`, `search_count` | no-regression: content search, newest file, counting |
| `term_wait_then_write`, `term_dev_server` | no-regression: fixed wait in the foreground, background server left running |

memory, skill_manage, search_files and terminal are the real tools. manage_connections is stubbed: the
real target normalisation and action validation run first (a wrong-side call is refused with the real
message), then a canned settled result comes back. Scores come from the stores, the workspace and the
recorded calls, never from model prose alone. Each family gets only the toolsets it needs plus
file/terminal as competition.

## Running

```bash
git worktree add .worktrees/main-snap origin/main
export SDD_BASE_TREE=$PWD/.worktrees/main-snap SDD_PR_TREE=$PWD
export SDD_PYTHON=<the install's committed env python>
export SDD_RESULTS=$HERMES_HOME/cache/scratch/schema_dedupe
python3 evals/schema_dedupe/run.py anthropic/claude-fable-5.2 3 --parallel=8
python3 evals/schema_dedupe/run.py --report
```

Inference defaults to the Nous Portal agent key, read fresh from `$HERMES_HOME/auth.json` for every cell
(read-only). Set `SDD_INFERENCE_URL` + `SDD_INFERENCE_KEY` for any other OpenAI-compatible endpoint.
Errored cells score 0 and stay in the denominator; exit 3 (no model turn, or a manage_connections call
that bypassed the stub) is an infra failure and is never written.

## Traps this harness already fell into

- **The agent loop runs manage_connections inline**, not through the registry handler. Stubbing only the
  registry entry leaves the real backend live: the first smoke minted real connect links. The seam is
  `tools.connectors.manage_connections`; the real backends are also replaced with a function that raises.
- **Nothing the model sees may carry the arm name.** With `base`/`pr` in the temp workspace path, two
  byte-identical tool surfaces behaved differently (one-call vs two-call on `term_wait_then_write`: 0/6
  vs 3/6). Cell directories are now named the same on both arms.
- **A seeded memory entry about tmux makes models start tmux.** `TMUX_TMPDIR` points inside the cell, and
  every process still running from the cell directory is killed by PID at the end.
- **A weak model can walk out of the workspace.** The sandbox moves HOME, it does not hide absolute paths:
  three qwen cells read the operator's real `~/.hermes` (read-only commands). Run from a scratch root that
  does not sit under the real Hermes home when that matters.
- **Dollar columns depend on provider cache warmth**, which depends on which arm's prefix was sent first
  and how recently. Compare tokens; treat dollars as the bill, not as an arm difference.

## Reference run

`results/2026-11-09/cells.jsonl` holds one row per cell (score, calls, token usage, cost, schema sizes);
transcripts are not checked in. Regenerate the tables with `SDD_RESULTS=<dir of per-cell json> run.py --report`.
`results/2026-11-09/round1-memory-pr.jsonl` is the PR arm's memory family from the first round, before
"Reusable procedures belong in a skill, not memory" was put back (qwen 42.5/45 against 45/45 on main).
