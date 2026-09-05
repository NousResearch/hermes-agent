# Wave-2 Blind Cluster-Map Pass (mapping stage recipe)

Proven recipe for a wave-1/wave-2 blind witness producing an independent cluster map of one godfile region (slack adapter R3 pass B, 2026-08-05). Complements `extraction-w2-*` (which adjudicate patches) — this is the **pre-extraction mapping pass** (pipeline stages 2–3). Blind rule: do NOT read sibling witnesses' analysis files; derive everything from live source.

## Sequence

1. **Locate the main repo + pinned worktree.** The main repo may be non-obvious (`C:/tmp/hermes-agent-verify` here, not a user dir). If unknown, read a sibling worktree's `.git` file: `cat C:/tmp/ws-<god>-r1/.git` → `gitdir: <main>/.git/worktrees/...`. Then:
   `git -C <main> worktree add --detach C:/tmp/ws-<god>-r<X>b <pin-sha>` (pass-B convention: `<god>-r<X>b`).
   Verify pin: `git rev-parse HEAD` == pin sha, `wc -l <godfile>` == recorded count (9,088 for slack adapter).
2. **Read the whole region** (read_file chunks), noting `# ----- Section -----` markers — the file's own seams.
3. **AST inventory with exact spans** (repo venv python):
   - `ast.walk` the module; collect `FunctionDef/AsyncFunctionDef` with `col_offset == 4` (class-body methods only — top-level-only scans miss class methods).
   - Print `lineno-end_lineno name` for every def whose span intersects the region window.
   - Fully-contained count + line sum via the boundary check: `region_start <= lineno and end_lineno <= region_end`.
   - **Count with the script, not by eye** — an eyeballed count was off by one (claimed 44/42, actual 43/41).
4. **Boundary-straddling check FIRST (highest-value finding).** Compare region start/end lines against method spans. Straddling methods are JOINT clusters owned by two region plans — record them as `interlocked_with` the adjacent region, never half-sliced. Slack R3 straddlers: `format_message` (3,550–3,707, head in R2) and `_handle_slack_message` (5,228–6,344, 884-line tail in R4). Every other method was fully contained in a contiguous 3,711–5,226 block — that contiguous-ness is what makes the region cleanly mappable.
5. **Dependency-edge enumeration.** For every helper referenced in the region, grep its `def` line and classify: in-region / other-region (which one) / module-level helper (stays). Then:
   - Intra-cluster calls prove cohesion (e.g. `_resolve_channel_name` → `_resolve_user_name`; `_resolve_user_is_bot` populates the name cache).
   - **Forward deps into LATER regions block standalone extraction of an otherwise-contained cluster**: wake machinery (`_should_wake_on_unmentioned_message`) calls `_fetch_thread_context` (R4), `_fetch_thread_parent_text` (R4), `_has_active_session_for_thread` (R5), `_slack_strict_mention` (R5). Such clusters are extractable only if those helpers stay in adapter or a joint multi-region plan exists.
   - Module-level helpers (e.g. `_extract_text_from_slack_blocks` at line 388) stay in the godfile — note that mixins may then need lazy `from <godfile> import` shims (cycle risk).
6. **Wiring/seam checks** (each is a cluster-selection criterion):
   - **Listener closures in an earlier region** reference the region's `_handle_*` via `self.` (e.g. Bolt `@app.event` registrations) — attribute lookup at call time → MRO-safe after mixin extraction; verify with a runtime identity probe later.
   - **Hook-protocol methods dispatched by string name** (`_run_processing_hook("on_processing_start", ...)` in base.py) — extraction must keep the exact names class-reachable (mixin MRO provides it).
   - **`super().X(...)` fallbacks** (e.g. `send_image` falling back to base) — mixin must precede the base class in MRO; treat as a (low) risk and an ordering constraint.
   - **Lazy imports** (`from tools.url_safety import ...` inside method bodies) must STAY lazy in the mixin — never hoisted to module level.
   - **ClassVars** (e.g. `_REACTION_EMOJI_MAP`) move with their owning cluster; tests may reference them via the class.
   - **`__init__`-owned instance state** (caches, `_dedup`, client maps) lives in R1 — mixins consume `self._*`, never re-init.
   - **getattr-guarded state reads** exist because tests build adapters via `object.__new__` without `__init__` — keep the guards.
7. **Test-surface mapping — scope greps to the platform's OWN test files.** Method names like `format_message`, `send_document`, `send_video`, `on_processing_start` are common across adapters: a repo-wide `grep -rln` matched 50+ telegram/discord/whatsapp tests, all false hits. Grep `tests/**/test_<platform>*` files only, count direct hits per file, group by cluster.
8. **Verify "no test coverage" claims by grepping the platform test files for the exact method names BEFORE writing them into the verdict.** This pass wrote "no dedicated reaction test file" into a risk item — deepening proved it wrong (`test_slack_approval_buttons.py` 584+ had a dedicated `_handle_slack_reaction` block; `test_slack.py` 2,553–2,569 had the primitives). Same class as the wave-3 "byte-count the artifact yourself" rule: claims about coverage must be checked, not assumed.
9. **Write the file immediately** (persist-early doctrine), verify >10KB (`ls -la`), then deepen and patch corrections into it. The deepening pass (seam probes, count corrections, coverage verification) produced three substantive patches to the already-written file.

## Verdict shape

- `EXTRACTABLE` / `EXTRACTABLE — with region-boundary coordination` / `BLOCKED`.
- First-extraction recommendation = smallest MRO-trivial cluster with existing test coverage and no super()/lazy-import/complex state (slack R3: C3 identity, 5 methods, 222 lines — zero super() calls, zero lazy imports, all state in `__init__` caches).
- Estimate scope: fully-contained method count + line sum (slack R3: 41 methods, 1,447 lines).
- Record blind-witness confidence: spans AST-verified, dep edges enumerated from live source, test mapping done, wave-1 outputs untouched.

## Evidence commands (all read-only, git-bash on Windows)

```bash
# straddle check + section markers
grep -nE "^    (async )?def |^    @staticmethod|^# ---|^class " <godfile> | awk -F: '$1>=<start> && $1<=<end>'
# dependency def lines for a batch of names
for n in a b c; do grep -n "def $n(" <godfile> | head -1; done
# platform-scoped test mapping
for f in tests/gateway/test_<platform>*.py; do echo "-- $f"; grep -cE "method_a|method_b" "$f"; done
```
