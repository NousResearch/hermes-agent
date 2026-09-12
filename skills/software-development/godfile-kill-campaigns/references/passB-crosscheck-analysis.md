# Wave-2 pass-B blind cross-check (slack wave, 2026-08-05)

Blind re-map of one godfile region (5th of 5 shards, lines 7281–9088 of a 9,088-line
adapter.py) with NO knowledge of pass-A. READ-ONLY except the single analysis
deliverable. Proven pattern for the wave-2 cross-check stage; every cluster claim
was later confirmed by the merge — the greps below are what made the report
verifiable.

## Deliverable discipline (write IMMEDIATELY, then deepen)
1. **WRITE the analysis file right after the full region read** — before any
   greps. The brief usually mandates a size floor (here: >10KB) and an exact
   path (`C:/tmp/tg-campaign/godfile/slack/R5-analysis-B.md`). Confirm with
   `wc -c`; a 20KB+ first pass leaves room to deepen.
2. **Deepen after writing** with cross-reference greps against the pinned
   worktree, then PATCH the file with a "verification receipts" section that
   cites every claim. The receipts turn assertions into evidence the wave-3
   validator can re-check.

## Fresh-worktree pin verification sequence (catches silent mangles)
```bash
# 1. Pin exists and file line count matches the brief AT the pin
git cat-file -t <PIN_SHA>                          # must print "commit"
git show <PIN_SHA>:plugins/platforms/slack/adapter.py | wc -l   # == brief's count

# 2. Worktree add with a NATIVE C:/ path — see pitfall below
git worktree add "C:/tmp/ws-<region>" <PIN_SHA>

# 3. Verify inside the fresh tree: HEAD == pin, line count matches
cd /c/tmp/ws-<region> && git rev-parse HEAD && wc -l <godfile>
```
If `cd` fails right after a successful-looking add, the tree went to a mangled
path — check `git worktree list`. **Fix (R4 pass-B, 2026-08-05)**: the add can
SUCCEED and register a mangled double-path tree (`C:/c/tmp/ws-sl-r4b` from an
MSYS `/c/tmp/...` arg) — remove with the mangled path form exactly as registered,
then re-add with the native form:
```bash
git worktree remove --force "C:/c/tmp/ws-sl-r4b"   # path as `git worktree list` shows it
git worktree add "C:/tmp/ws-sl-r4b" <PIN_SHA>
git worktree list | grep <region>                  # confirm native path registered
```

## Pass-B cluster-map evidence pattern (per cluster)
- **Members table**: every member with exact line range + role. Group methods
  that share state (`_thread_context_cache`, watermark keys, ContextVars) into
  one cluster even when not contiguous (contiguous spans + shared-state spans
  both count; note non-contiguity explicitly).
- **Coupling edges**: for each cluster, list (a) instance state it reads,
  (b) external modules, (c) which OTHER clusters consume/produce it, (d) callers
  in earlier regions. The directional edge count is the cleanest-slice signal:
  a cluster with zero inbound-from-cluster and only config/env reads is a leaf.
- **Banner-delimited contiguous blocks** (explicit `# ── ... ──` comments) are
  the mechanically cheapest cuts — call them out.

## Receipt greps that made the report verifiable
```bash
# Consumers of a cluster's methods in OTHER regions (proves seam is call-site-zero-edit)
grep -n "_slack_require_mention\|_slack_strict_mention\|..." adapter.py | grep -v "def _slack"

# Cross-region shared state (ContextVar, caches, stash dicts) — must MOVE with the cluster
grep -n "_slash_user_id\|_slash_command_contexts" adapter.py     # def site vs consume site
grep -n "_thread_context_cache\|_ThreadContextCache" adapter.py  # cache shared across shard boundary

# Module-level import sources (which gateway.base / agent helpers back the cluster)
grep -n "^from\|^import" adapter.py | head -50

# Test coverage PER CLUSTER (34 slack test files; count references per member)
for f in tests/gateway/test_slack.py tests/gateway/test_slack_mention.py ...; do
  echo "== $f"; grep -o "_handle_slash_command\|_standalone_send\|..." "$f" | sort | uniq -c; done
```
Key receipts from the slack run: C4 gate-policy consumers all at 5223–5755
(`_should_wake_on_unmentioned_message`), ContextVar def at 260 + consume at
1483–1517 inside `send()`, `_thread_context_cache` init at 985–988 + reads at
5113–5130 and 7209–7217 (cache spans the R4/R5 shard boundary).

## Cleanest-slice scoring (first-extraction recommendation)
Rank by: purity (no async/I-O/adapter calls), contiguity (banner-delimited),
precedent (mirrors an existing platform mixin, e.g. DiscordGateMixin #79653),
seam blast radius (call sites unchanged), test leverage. The slack verdict:
C4 gate/mention policy (8231–8464) → `SlackGateMixin` first; C3 file-download/
SSRF second; C1 thread-context last (deepest entanglement).

## Risk patterns worth naming in every pass-B report
- **Env-name contract**: a YAML→env bridge in the tail writes the exact
  `SLACK_*` vars the gate getters read — cross-check the two key sets; recommend
  a test that pins the contract.
- **`__new__`-without-init borrow hack**: a standalone-send helper instantiates
  the adapter class to borrow `format_message` — works only while that method is
  init-independent; recommend hoisting before extracting.
- **Private-store pokes**: `_has_active_session_for_thread` reads
  `session_store._entries` / `_ensure_loaded()` — fail-closed wrapper needed if
  the store refactors.
## Security invariants**: SSRF/CDN-allowlist/redirect-revalidation guards must
be called out as non-negotiable in extraction (regression there is a security
bug, not a refactor issue).

## R4 pass-B additions (slack R4 = 5461–7280, 2026-08-05)

The R4 window's shape differs from R5: the first ~880 lines are the TAIL of a
giant straddler from the earlier region (`_handle_slack_message` def @5228 in
R3's window), then 11 R4-owned methods. Techniques that made this report solid:

- **Straddler-at-start rule**: a method whose `def` line falls in an EARLIER
  region's window belongs to that region — the later region must NOT slice it,
  no matter how much of the region its body physically occupies. Record the
  straddler's full span for the earlier lane (R3 must know its tail reaches
  6344) and treat the region's own content as starting at the next section
  comment (`# ----- Approval button support (Block Kit) -----` @6346). Extract
  only between that comment and the next straddling method's start.
- **`super()` census → mixin-first is non-negotiable**: grep `super(` inside
  the proposed move window before recommending. Exactly ONE live `super()`
  call (`send_clarify` @6562 delegating to `BasePlatformAdapter`) makes the
  class line `class SlackAdapter(SlackInteractiveMixin, BasePlatformAdapter)`
  MANDATORY — mixin AFTER the base would resolve to `object` and raise
  AttributeError on that path. Any cluster containing a `super()` call needs an
  MRO assertion in the verification battery (`SlackAdapter.__mro__[1] is
  <Mixin>` + `SlackAdapter().send_clarify.__func__ is <Mixin>.send_clarify`).
- **Lazy in-method `from tools...` imports travel cleanly**: `_handle_*_action`
  methods import `tools.approval`/`tools.slash_confirm`/`tools.clarify_gateway`
  INSIDE the method body (local import @6911 etc.) — these move with the method
  with zero adapter coupling; verify the primitive's def site in the tools
  module before claiming "travels with" (grep `def resolve_gateway_approval`
  in tools/approval.py). Distinguish these from adapter-owned module helpers
  (e.g. `_extract_text_from_slack_blocks` @388) that stay behind because other
  regions use them — those need lazy-import seams in the mixin.
- **Open-PR census via `gh api search/issues`, not PR-by-PR diffs**: the
  pass-A "one PR touches the file" note went stale. `gh api search/issues -f
  "q=repo:<owner>/<repo> is:pr is:open <file>"` returns the full open-PR set
  touching the godfile (50 for slack/adapter.py). Categorize by which
  cluster/window methods each PR's title/description targets (clarify-choice
  PRs → C1 senders/handlers; thread-context retry PRs → C3), and state the
  mandatory re-scan-before-ship explicitly in the risks table. Bodies
  mentioning the filename inflate the count — treat it as a superset to
  categorize, not a precise diff map.
- **Citation precision: grep the ASSIGN site, not the first grep hit**: a
  first-pass citation for `_app` pointed at @2286 (`self._app = None` — a
  disconnect RESET site); the real assign is @1901 (`self._app = AsyncApp(...)`
  in connect). Rule: when citing where an attribute is SET, grep all `self.X =
  ` lines and cite the initializing one, and say what the reset sites are
  (both are extractability-relevant: init stays, reset stays, only reads move).
  Same for constants: `_THREAD_CACHE_MAX` is an instance attr init @988, not a
  class const — verify before writing "class const" in the report.
- **Instance-state contract check**: before recommending a mixin, list every
  `self._X` read in the window and confirm its init site is OUTSIDE the window
  (init @952–957, `_app` @1901) → zero `__init__` edits needed. That was the
  C1 cluster's cleanest signal (state stays on the adapter, methods only read).
- **Verification battery as a table**: a "Check | Command | Result" table
  (pin identity, file length, clean worktree, AST inventory count, handler
  wiring refs, external callers, module deps, super() census, test surface,
  open-PR scan) makes every claim independently re-checkable — the wave-3
  validator's entry point. Write it in §6, immediately before the verdict.

## R1 pass-B additions (hermes_cli/main.py R1 = 1–2520, 2026-08-05)

The R1 window is the module HEAD of a CLI godfile: ~59 top-level defs, zero
classes, and heavy import-time executable blocks (the startup-order contract).
Techniques that made this report solid:

- **End-straddler rule (mirror of the R4 start rule)**: a method whose `def`
  line falls in YOUR window but whose body crosses the region boundary belongs
  to your region — the NEXT lane must not slice the tail, no matter how short
  it is. Record the exact tail span for the next lane from AST
  `node.end_lineno`, never eyeballed: `_resolve_use_tui` def @2485, span
  2485–2525, so R2 was told "do NOT slice 2521–2525" (a 5-line tail). The R4
  rule covers def-in-earlier-region; this is the def-in-own-region case and
  the recording duty is symmetric.
- **First-pass def inventories are approximate — the receipts table is where
  they get corrected to AST truth**: the write-immediately discipline means §1
  ships an eyeballed count (48 here) that the deepen pass corrects to the AST
  count (59). Two classic undercount sources: nested defs inside a parent
  (`_inside_mcp_add_args`/`_resolve_sudo_user_profile_env` live INSIDE
  `_apply_profile_override` — not top-level, must not appear in the top-level
  inventory) and dense one-line proxy stubs. Correct the §1 table to AST
  `(name, lineno, end_lineno)` per def before finalizing; the receipt cites
  the script, not the eye.
- **Open-PR census precision — hunk ranges, not titles**: title/body-based
  categorization is a superset (912 raw hits for main.py). Decide per-region
  overlap by extracting the actual old-side hunk ranges:
  `gh pr diff <n> | awk '/^diff --git a\/<godfile>/{f=1;next} /^diff --git/{f=0} f && /^@@/{print $2}'`
  → `-388,11` means PR #73455 touches lines 388–398 (C2). This yields the
  decisive claim "the recommended window (105–221) is collision-free" plus a
  per-PR cluster map. State the mandatory re-scan-before-ship explicitly.
- **Reverse-deps + lazy-import re-export contract**: before recommending an
  extraction, grep who imports the godfile:
  `rg -l "from hermes_cli.main import|import hermes_cli.main" hermes_cli --glob "*.py"`.
  Zero TOP-LEVEL importers ⇒ no circular-import risk, but every extracted name
  must stay resolvable as `hermes_cli.main.<name>` because all consumers are
  lazy in-body imports (129 test files reference `hermes_cli.main` here). Then
  find the DECISIVE regression test for the seam — the one that imports the
  moved name directly (`from hermes_cli.main import _exit_after_oneshot` in
  test_tui_resume_flow.py) — and name it in the seam/test plan as the w3 gate.
- **Windows grep tooling: `search_files` mangles worktree paths, terminal rg
  does not**: search_files on `C:/tmp/ws-mn-r1b` failed with
  `rg: /c/tmp/ws-mn-r1b: IO error ... (os error 2)` (MSYS path mangling inside
  the tool), while terminal `rg -n <pat> C:/tmp/ws-mn-r1b/...` with native
  `C:/` paths worked fine. Run worktree greps via terminal rg with native
  paths; same class as the git-worktree-add mangling pitfall, different
  surface.

## R5 pass-B additions — CLI godfile tail (main.py 10081–12599, 2026-08-05)

A CLI godfile tail is a DIFFERENT shape from an adapter tail: it is the dispatch
layer — thin `cmd_*` delegators, launch-routing helpers, and one giant
parser-construction `main()` — not a method farm over shared adapter state.
Techniques that made the main.py R5 report solid:

- **Definition inventory FIRST, by script**: exactly 37 `def` statements in the
  window (31 module-level: 17 `cmd_*` + 13 private helpers + `main`; 5 nested
  inside `main()`; plus 1 lambda) — count with
  `awk 'NR>=A && NR<=B && /^def |^class |^    def /' file | sort | uniq -c`,
  then measure spans with a tiny python loop (def → next top-level `def`).
  Counting by eye on a 2,500-line tail is how false "41 moved" claims happen.
- **In-file vs IMPORTED helper resolution — the critical seam test**: a helper
  the region references may be defined in an EARLIER region of the same file
  (→ cross-region edge; it must move first or be imported back, creating a
  main→newmodule→main cycle) OR already imported from a sibling module (→ the
  seam ALREADY EXISTS; the mover only re-imports it). Resolve with
  `grep -n "^def X" godfile` vs `grep -n "import.*\bX\b" godfile`. main.py R5:
  15 helpers defined earlier in-file (`_run_and_exit_oneshot` @176,
  `_build_web_ui` @5606, ...), but `cmd_sessions` and
  `_kill_stale_dashboard_processes` are already imports. A "helper must move
  too" claim without this check is the classic extraction-plan overreach.
- **Decomposition precedent detection**: before recommending a new module, grep
  the godfile for commands ALREADY extracted (`from hermes_cli.sessions_cmd
  import cmd_sessions` @439) — it reveals the house wiring pattern. main.py's
  precedent: move the handler → keep parser construction in `main()` → thread
  via `functools.partial(cmd_sessions, sessions_parser=…)` when the handler
  needs a `main()`-local parser (12364 comment: "main.py decomposition").
  New extraction modules should follow the established precedent, not invent
  wiring.
- **Nested-closure census inside the parser-construction function**: before
  recommending any move out of a 1,400-line `main()`, enumerate the defs nested
  inside it and which `main()`-locals each captures (`_dispatch_secrets`,
  `_dispatch_egress`, `cmd_computer_use`, the completion lambda) — naive
  verbatim moves → `NameError`. Capture-free nested defs (`cmd_import_agent`,
  `_add_session_filter_args`) move plainly; closure-capturing ones need hoisting
  with explicit parser parameters or stay nested. This is the "main() last"
  risk, provable in the report.
- **Cleanest-slice scoring for CLI tails**: rank by closure-freedom + leaf-ness
  + sibling-module existence, not just purity/contiguity. main.py verdict:
  C2 thin delegators (leaf `cmd_*` bodies, zero closures, zero global state,
  each already delegating to a split sibling module) FIRST; C1 launch-routing
  helpers second (pure, but many cross-region call edges: `cmd_chat`,
  `_run_and_exit_oneshot`); C3 dashboard cluster third (8 external helper
  deps); C4 `main()` LAST.
- **Command-handler extraction compatibility trap**: subcommand builders in
  `hermes_cli/subcommands/*.py` receive `cmd_*` callables at `main()` call
  time (`build_dashboard_parser(subparsers, cmd_dashboard=cmd_dashboard)`), so
  as long as `main.py` binds the names before `main()` runs, nothing else in
  the repo changes — **name-binding IS the seam** for CLI tails.
- **Parity contracts need a mechanical test BEFORE extraction**:
  `_BUILTIN_SUBCOMMANDS` (frozenset of subcommand names) must stay in sync with
  the `add_parser` calls, and `_TOP_LEVEL_VALUE_FLAGS` with `_parser.py` flags
  (explicit sync comments at 10600/10626 warn of drift). Add a test that walks
  `subparsers.choices.keys()` after a real parser build and asserts
  set-equality BEFORE any parser-registration reorganization, so drift is
  caught mechanically instead of as a silent fast-path gate regression.
- **Security-critical verbatim-move rules for CLI tails**: `--yolo` ordering
  guarantee (HERMES_YOLO_MODE set before `tools.approval` import, PR #7994 —
  chokepoint at `_prepare_agent_startup`) and fd-traversal token readers
  (`_read_ssh_session_token_file`: O_NOFOLLOW/O_DIRECTORY, owner/permission
  checks, unlink-on-read) must be extracted VERBATIM, zero reformat — same
  class as the adapter SSRF guards; call them out as non-negotiable in the
  risks table.

## R2 pass-B additions — the PRIOR-WAVE COLLISION MAP (kanban_db R2 = 2061–4120, 2026-08-05)

The wave-1 blind analysis of kanban_db.py's R2 discovered the godfile had ALREADY
been killed once: five open unmerged PRs (#79613–#79617) from an earlier kanban
wave, whose shard-s2 PR (#79614) extracted exactly the feature-CRUD half of the
region under analysis. Techniques that made the report decisive:

- **Run the open-PR census DURING wave 1, not just pre-ship**: `gh api "search/issues?q=repo:<owner>/<repo> is:pr is:open <godfile>"` at analysis time, then per-PR hunk ranges (`gh pr diff <n> | awk '/^diff --git a\/<godfile>/{f=1;next} /^diff --git/{f=0} f && /^@@/{print $2}'`). Decode each `-old,count` hunk into an old-line range and intersect it with each cluster's AST window → a per-cluster FREE/COLLIDED verdict table. #79614's hunks `-2869,475 -3353,58 -3521,116 -3701,395` pinned six of fourteen clusters as collided; the gaps between hunks (3411–3520, 3637–3700) marked the clusters the prior witness left behind.
- **Re-scope the first-extraction recommendation to the collision-free remainder**: the naive best slice (attachments — dedicated test file, cleanest banner) was fully inside a prior PR's window; the final pick (txn primitives 2731–2838) was collision-free AND the exact part the prior wave skipped. Verdict shape: "retarget/credit #79614, never re-extract" + a new slice order built from the free clusters (C4→C5→C3→C10→C14). A prior wave's extraction is live context for the new wave, not a problem to ignore.
- **Mis-titled sibling-PR detection by FILE LIST, not title**: #79659 carried a kanban_db title ("extract txn/task-link mixins (shard s2)") but `gh pr diff 79659 | grep "^diff --git"` showed main.py/aux_config_cmd.py/model_picker.py and ZERO godfile hunks — it is the OTHER god's (main.py) kill wearing a sed-derived kanban title. Titles lie; the diff file list is ground truth. Same class as the tui sed-derivation mis-title pitfall, verified at PR level.
- **Shared bottom re-export tail is its own interlock**: every shard PR in a multi-shard kill rewrites the same bottom re-export block (`-10273,3` appeared in all five kanban PRs) — that tail is a cross-PR collision the merger must sequence even when the shard windows are disjoint. Record it as a row in the collision table.
- **Boundary-ambiguous hunk**: a prior hunk starting on a def's LAST line (`-3521,116` = `set_reasoning_effort`'s final line) makes the adjacent cluster boundary-ambiguous — verify the exact first deleted line against the real diff before slicing that cluster; a one-line straddle changes the window by one def.
- **Functional-module godfile shape**: zero classes-with-methods, zero `super()` → the extraction target is a NEW MODULE + bottom re-export block (never a mixin class); consumers use `from hermes_cli import kanban_db as kb` so the re-export preserves `kanban_db.<name>` with zero call-site edits. Before recommending, grep consumers for DIRECT by-name imports of private helpers (`plugin_api.py` imports `_safe_attachment_name`/`_collision_free_path` from the godfile) — those names must ride the re-export block or the consumer gets patched (a wiring edit flagged in the PR).
- **Golden sha receipt pair**: record BOTH `git show <pin>:<godfile> | sed -n 'a,bp' | sha256sum` (no trailing NL) AND `| git hash-object --stdin` (blob sha1) for the recommended window, with the actual values inlined into the receipts section — the wave-3 validator matches either form.
