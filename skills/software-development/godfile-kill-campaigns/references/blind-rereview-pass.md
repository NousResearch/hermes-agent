# Post-commit blind re-review pass (discord + slack waves, 2026-08-05; pass A golden-sha, pass B adversarial)

READ-ONLY re-review of committed-but-unpushed god-file kill slices against their
`*-CONSENSUS.md` contracts. Proven on all 5 discord slices (`9487b841be`,
`d7873a3504`, `518cc6f27b`, `b5f7ce2541`, `d062cc1607`; branches
`fix/dc-standalone-send|gate-mixin|media-mixin|voice-receiver|recovery-backfill`)
and the slack R1/R5 slices (pass A, 243cd0254d / e69179663a — both APPROVED;
golden-sha variant matching + `/tmp` path trap in Setup §4).
Verdict per slice: APPROVED / REQUEST_CHANGES (CRITICAL/IMPORTANT/MINOR), evidence-cited.

## Setup
```bash
# 1. Confirm each worktree at its slice HEAD, clean, correct branch
git -C "C:/tmp/ws-dc-<slice>" rev-parse HEAD      # native C:/ paths — MSYS /c/ rejected
git -C "C:/tmp/ws-dc-<slice>" branch --show-current
git -C "C:/tmp/ws-dc-<slice>" status --porcelain  # empty

# 2. Extract the pin godfile once (all slices share the pin)
git -C C:/tmp/ws-dc-<slice> show <PIN>:plugins/platforms/discord/adapter.py > C:/tmp/pin_adapter.py
wc -l C:/tmp/pin_adapter.py                        # must equal consensus line count (10,138)

# 3. Golden blob gate: diff pre-image must BE the pin blob
git diff HEAD~1 HEAD -- <godfile> | grep '^index'  # e.g. index 22d4c2af2d.. == consensus sha prefix
git rev-parse <PIN>:<godfile>                      # full sha must equal consensus LF-normalized blob

# 4. Golden sha gate (the brief's golden sha covers a VARIANT slice — locate it):
# try window-only vs window+banner, with/without trailing NL, sha256/raw sha1/git-blob sha1.
# A "pin-slice + '\\n' == module-slice" hit is the byte-identity proof (module slice carries
# the file's EOF newline via split(b"\\n")'s trailing "" element).
python - <<'EOF'
import hashlib
pin = open(r"C:/tmp/pin_adapter.py", "rb").read()
plines = pin.split(b"\n")
mod = open(r"C:/tmp/ws-sl-<slice>/plugins/.../<new_module>.py", "rb").read()
mlines = mod.split(b"\n")
# candidates: window-only and banner+window, each ± trailing NL; also try
# git-blob format: hashlib.sha1(b"blob %d\\0" % len(b) + b).hexdigest()
def sha(b, algo=hashlib.sha256): return algo(b).hexdigest()
win_no_nl  = b"\n".join(plines[141:252])            # R1 slack: matched golden b2c71c1e…
win_nl     = win_no_nl + b"\n"                      # R5 slack: matched golden 8a157a20…
EOF
# NOTE: git-bash `> /tmp/x` writes to MSYS temp (C:\Users\<u>\AppData\Local\Temp); native
# Windows Python cannot open /tmp/x or C:/tmp/x — cygpath -w /tmp/x first, or hash in bash.
# Module docstrings may record a COMMIT sha as "Pin sha" (slack R5 said 1be70d63… while the
# real pin blob was ee50a4578c) — always verify the blob claim via git rev-parse, don't trust
# the docstring label.
```

## 1. Byte-verbatim move vs pin (AST slice compare)
- Collect moved members from the **pin window**, never from the task brief:
  parse pin with `ast`, find the `DiscordAdapter` class node, take every
  FunctionDef/AsyncFunctionDef/Assign/AnnAssign whose `lineno` falls inside the
  consensus window (e.g. R4 = 6116–6465 → exactly 28 members). Top-level-only
  `tree.body` scans MISS class methods — walk the class body for mixin slices.
- Slice each member with `"\n".join(src.split("\n")[node.lineno-1:node.end_lineno])`
  (instant; `ast.get_source_segment` is O(n²) on 10K-line files).
- Compare pin-slice vs module-slice. Sanctioned differences only: lazy adapter
  imports (see normalizer below). Anything else = DIFF finding.
- **adapter_clean**: same members must be ABSENT from the new adapter.py
  (class-body aware). Presence = collateral.
- **Mutation-sensitivity positive control**: mutate one member in-memory
  (`src.replace("def X(self):", "def X(self):  # MUT")`), re-run the slice
  compare, assert it flags DIFF. Proves the checker isn't vacuous.

### Lazy-import normalizer (REMOVE, not blank)
Insertion shapes observed: R3 = import alone mid-body (after `async with`),
R4 = import + trailing blank line at body top. Both absolute
(`from plugins.platforms.discord.adapter import X`) and relative
(`from .adapter import X`) forms appear. Skip the import line AND the
immediately-following blank when present; never eat a blank after a non-import
line; never strip non-adapter imports (over-strip guard).

```python
def strip_lazy_adapter_imports(body):
    out, skip_next_blank = [], False
    for ln in body.split("\n"):
        s = ln.strip()
        if (s.startswith("from plugins.platforms.discord.adapter import")
                or s.startswith("from .adapter import")
                or s.startswith("from ..discord.adapter import")):
            skip_next_blank = True
            continue
        if skip_next_blank:
            skip_next_blank = False
            if ln.strip() == "":
                continue
        out.append(ln)
    return "\n".join(out)
```

## 2. Seam identity
- Dedicated seam tests (per-slice files listed below) assert
  `getattr(DiscordAdapter, n) is getattr(Mixin, n)` for every moved member and
  `DiscordAdapter.__mro__[1] is <Mixin>` — run them.
- Cheap runtime probe complement:
  `sys.modules.setdefault("discord", MagicMock())` then per-member identity
  compare; for module-level classes (R1 VoiceReceiver, R5 standalone fns):
  `adapter.VoiceReceiver is voice_receiver.VoiceReceiver` /
  `getattr(adapter, n) is getattr(standalone_send, n)`.
- R5: `register()` must still wire `standalone_sender_fn=_standalone_send`
  through the re-export — `inspect.getsource(register)` and compare the body
  against the pin (byte-identical expected).

## 3. No module-level adapter import
```bash
grep -nE "^(from|import) .*adapter" <new-module>     # must be empty
grep -cE "from (\.|\.\.|plugins\.platforms\.discord\.)adapter import" <new-module>
# count = lazy in-method imports only (0 for R1/R2/R5, 2 for R3, 3 for R4 — sanctioned)
```

## 4. Class line
```bash
grep -n "^class DiscordAdapter" <godfile>
```
- R2/R3/R4: mixin-FIRST (`class DiscordAdapter(<Mixin>, BasePlatformAdapter):`) —
  consensus-mandated for all three (base stubs would win MRO otherwise).
  Adjudicate against the CONSENSUS doc, not the task brief (brief said "R3
  appended"; R3-CONSENSUS mandates mixin-first — implementation + `__mro__[1]`
  probe agree with consensus).
- R1/R5: module-level class/functions — class line unchanged.

## 5. Zero collateral (adapter.py diff = import + class line + window only)
```bash
# Deleted ranges via -U0 hunk parse: every consensus-window line covered;
# allowed extras EXACTLY: import lines, class line, trailing blank boundary.
git diff -U0 HEAD~1 HEAD -- <godfile> | grep -E '^@@'
# All added lines must be import/class-line only:
git diff HEAD~1 HEAD -- <godfile> | grep '^\+' | grep -v '^+++'
```
- Trailing-blank boundaries at window ends are expected (e.g. 898/2746/6466/9798).
- R1 also removes now-dead imports (`env_int` unused pin-wide; `threading`/
  `defaultdict` used only inside the moved window) — verify with a pin-wide
  usage scan before blessing an import removal.

## 6. Tests
- Seam tests: `tests/gateway/test_voice_receiver_seam.py` (R1),
  `tests/plugins/platforms/test_discord_recovery_backfill_seam.py` (R2),
  `tests/plugins/discord/test_discord_media_send_seam.py` (R3),
  `tests/plugins/platforms/test_discord_gate_mixin_seam.py` (R4),
  `tests/plugins/discord/test_discord_standalone_send_seam.py` (R5).
  Locate via `git show HEAD --name-only | grep tests` — paths vary and the
  diff-stat `.../` ellipsis hides them. New test dirs need `__init__.py`.
- Pre-existing per-slice suites (unchanged by the commit — verify the commit
  touches ONLY the seam test): R1 `test_voice_command.py` +
  `test_discord_race_polish.py` + `tests/integration/test_voice_channel_flow.py`
  (marker `pytestmark = pytest.mark.integration` — pass `-m ""` to run); R2
  `test_discord_missed_message_backfill.py`; R3 `test_discord_send.py` +
  `test_send_image_file.py` + `test_send_multiple_images.py` + `test_send_retry.py`
  + `test_73771_media_resend_dedup.py`; R4 the 11 gate files + `test_discord_gate_isolation.py`
  (in `tests/plugins/platforms/`, NOT `tests/gateway/`); R5 `test_send_message_tool.py`
  + `test_discord_send_message_caption.py` + `test_discord_tool.py` +
  `test_send_message_missing_platforms.py` + `test_discord_send.py`.
- Run with repo venv python, `--basetemp <outside-worktree> -p no:cacheprovider`
  (Windows phantom RC=1 quirk).
- **Baseline-verify every failure**: `git worktree add C:/tmp/<name>-base HEAD~1`
  (HEAD~1 = pin base) and re-run the failing test there. Identical failure =
  env-dependent. Discord R3: `test_73771_media_resend_dedup.py::test_streamed_explicit_media_resend_is_delivered`
  asserts a raw Windows path inside URL-encoded `file://C%3A...` — fails
  identically at pin. Remove base worktree after (`git worktree remove --force`
  + `git worktree prune`).

## 7. Hygiene
- DCO: `git log -1 --format="%b" HEAD | grep Signed-off-by` present.
- `git diff --check HEAD~1 HEAD` clean.
- `ruff check <new-module> <godfile>` clean.
- LF-only: **Python byte counts, NOT grep `$'\r'`** (MSYS mangling gave 0 vs 303
  for the same file):
  ```python
  data = open(p, "rb").read()
  crlf = data.count(b"\r\n"); lone = data.count(b"\r") - crlf
  ```
- Import OK: `python -c "import plugins.platforms.discord.adapter"` per slice.
- Not pushed: `git branch -r --contains HEAD | wc -l` == 0.
- No junk committed: `git show HEAD --name-only | grep -iE "__pycache__|\.pyc|basetemp"` empty.

## Result shape
Per slice: APPROVED / REQUEST_CHANGES with per-contract-item evidence (line
numbers, test counts, sha matches). All 5 discord slices passed every item:
83/83 members byte-verbatim (modulo sanctioned lazy imports), zero collateral,
116 seam tests + 256 pre-existing tests green (1 env-dependent failure
baseline-verified), DCO/ruff/diff-check/LF/import clean, nothing pushed.

## Slack wave pass B (adversarial re-review, 2026-08-05) — R1 243cd0254d + R5 e69179663a, both APPROVED

Both slices were clean 3-file commits (godfile + new module + seam test); the
adversarial pass added these traps:

- **Mutation positive control: the replace string MUST match the real def
  signature.** `src.replace("def _slack_require_mention(self):", "… # MUT")`
  does NOT match the live `def _slack_require_mention(self) -> bool:` — the
  replace silently no-ops, the control reports `flagged=[]`, and the checker
  looks vacuous. That is a false alarm on the CHECKER, not the code. Fix:
  copy the def line verbatim from the module (`grep -n 'def <name>'`), and when
  a control flags nothing, first assert the mutation landed
  (`'# MUT' in mutated_src`) before believing anything. Both slack controls
  (R1 7/7, R5 12/12) then flagged exactly the mutated member.
- **Runtime probes across TWO worktrees of the same repo: never
  `sys.path.insert(0, w1); sys.path.insert(0, w2)` in one process.** The
  index-0 tree shadows the other: `plugins` resolved from R5's tree and R1's
  slice-only module (`markdown_table_helpers` exists only in R1's tree) died
  with `ModuleNotFoundError` — a probe bug, not a code bug. Fix: one
  SUBPROCESS per worktree, `subprocess.run([py, '-c', probe], cwd=<wt>)` —
  `python -c` puts cwd at sys.path[0], so `import plugins...` resolves to that
  tree. (The seam-test suites are immune: pytest's rootdir insertion resolves
  the cwd tree.)
- **Name-prefix "no leftover defs" scans false-flag sibling clusters.**
  "Zero `def _slack_*` remain in adapter.py" is only true for the MOVED set:
  slack R5 legitimately keeps 8 other `_slack_*` methods owned by OTHER
  clusters (allow_bots, dedup_ttl_seconds, file_marker, ignored_channels,
  mention_detection_text, reaction_trigger_target, reaction_triggers,
  timestamp_sort_key — pin lines 122–4981, untouched by the diff). Scope the
  check to the moved member names (or prove via `git diff -U0` hunk coverage:
  R5 = 3 hunks / 3 added lines: import, blank, class line), and report prefix
  matches outside the moved set as "expected leftovers".
- **Adjudicate docstring pin-sha claims with git objects, don't assume
  staleness.** R5's mixin docstring claims "Pin sha: 1be70d6354…" while the
  commit base blob is ee50a4578c. `git cat-file -t` showed 1be70d6354 is a
  COMMIT, and `git rev-parse 1be70d6354:<godfile>` == ee50a4578c — the claim
  was accurate (consensus pin commit whose adapter blob is byte-identical to
  the base). Verify before flagging; complements the pass-A note about
  docstring shas.
- **Expected slack diff shapes.** R1 (module-level function slice): re-export
  at the BOTTOM of adapter.py (`from .markdown_table_helpers import (  # noqa: E402,F401`
  + names) — safe because the only caller (`format_message` line 3454) runs at
  RUNTIME; verify no module-level statement ABOVE the re-export uses the names
  at import time. One dead import removed (`import unicodedata`, zero uses
  outside the moved window in the pin — pin-wide usage scan). R5 (mixin
  slice): `from .slack_gate_mixin import SlackGateMixin` placed MID-FILE
  immediately before the class line (line ~854, not in the top import block);
  class line `class SlackAdapter(SlackGateMixin, BasePlatformAdapter):` at 857;
  deletion hunk = banner + 12 members + trailing blanks (236 lines). ruff
  passes because the re-export carries `# noqa: E402,F401`; the mid-file import
  needs no noqa (it precedes the class line it serves).
- **Seam-test filename may not contain the platform name**: slack R1's seam
  file is `test_markdown_table_helpers_seam.py` (no "slack") — `ls | grep -i
  slack` misses it and looks like a missing file. Verify via `git show HEAD
  --name-only` or a full `ls`, never a name grep.
- **Pre-existing gate-behavior suites for R5-type slices live in SEPARATE
  files, not test_slack.py**: `tests/gateway/test_slack_ignore_other_user_mentions.py`,
  `test_slack_mention.py`, `test_slack_require_mention_channels.py`,
  `test_slack_peer_agent_smoke.py` — 42 passed against the mixin tree (the
  `grep -nE 'def test_.*(mention|gate|…) tests/gateway/test_slack.py` probe
  finds ZERO because the gate tests aren't in that file). Cross-check the
  module docstring's window claim (R5 claimed 8233–8464; AST min/max lineno
  matched exactly) — docstring windows are reliable cross-check evidence when
  the AST span agrees.

## Slack R2-S1 pass A (d40b72686f, 2026-08-05 — APPROVED)

Messaging-family slice: 7 members, window 2439–3023 (585L contiguous), mixin
`plugins/platforms/slack/messaging_mixin.py` `SlackMessagingMixin`. Passed
every contract item (7/7 verbatim, identity, mixin-FIRST MRO, +3/-586, seam
test 15 + 197 pre-existing, hygiene). Three nuances beyond the R1/R5 recipes:

- **Full-window sha gate catches inter-member spacing that per-member slices
  miss**: per-member AST slices don't cover the blank lines BETWEEN members.
  After the per-member verbatim check, sha-compare the WHOLE pin window
  against the mixin's class body (class line +1 .. EOF): `window + b"\\n" ==
  body_without_trailing_nl` matched (sha256 `20e7b891…` both sides) — that
  proves the 585L body including separator blanks is byte-identical. Expect
  slice-arithmetic bookkeeping to print confusing `False` lines (pin file via
  `git show > f` then `split(b"\\n")` drops the trailing `""`, so window-no-NL
  ≠ body-no-NL and window+NL ≠ body+NL): the decisive comparison is
  `window+NL == body-no-NL` plus matching sha VALUES; per-member AST equality
  stays the primary evidence, never read the two `False` prints as findings.
- **Deletion hunk can equal the consensus window EXACTLY — trailing-blank
  boundaries are not universal**: `-U0` hunk was `-2439,585 +2440,0` — zero
  extras, not even the trailing blank the R5 pass documented as expected.
  Adjudicate against the consensus's stated window (585L here), not the
  pass-B expectation that blank boundaries always ride along.
- **Consensus dep lists and import-block suggestions are aspirational**:
  consensus §5 listed `asyncio` among mixin deps; the module correctly
  omitted it (zero `asyncio.` refs in the moved bodies — grep before
  flagging an omitted import). Consensus suggested the mixin import go in the
  adapter's top @45–60 block; it landed at line 67 right after the
  `block_kit` try/except — functionally equivalent, ruff-clean (no E402), no
  cycle. Adjudicate on body-reference reality + ruff-clean + no-cycle, not
  literal consensus wording.
