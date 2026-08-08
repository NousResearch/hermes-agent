# Wave-2 pass (blind cross-check B) — evidence recipe

How to run one blind cross-check pass (pass B / wave 2) on a godfile region
when handed: a region line range, a scratch worktree path "at pin", and a
target output file path. READ-ONLY except the single analysis file.

## Bootstrap when no local repo exists
The task may name a "scratch worktree at pin (C:/tmp/ws-...)" while no parent
clone exists on the machine. Search first
(`find <home> -maxdepth 5 -type d -name .git`, also check other drives), then
use a shallow clone as the worktree substitute:

    git clone --depth 1 --branch main <url> <scratch-path>

Verify the pin: `git rev-parse HEAD` inside the clone, and confirm the
godfile's line count matches the stated total (e.g. 10275 lines).

## Integrity check that matters (Windows shallow-clone quirk)
After a shallow clone on Windows/git-bash, `git status` frequently shows the
same file as BOTH `D` (deleted) and `??` (untracked) — clone-time checkout
artifacts, often thousands of entries; the clone even prints
"retry with 'git restore --source=HEAD :/'". Do NOT trust `git status`.
Verify the file you read IS the pinned blob:

    git rev-parse HEAD:<path>   # blob at pin
    git hash-object <path>      # blob on disk
    # equal ⇒ byte-identical to origin/main — cite that in the report

## Read the region, then WRITE IMMEDIATELY
1. Read the whole region in ≤2000-line chunks (read_file caps ~100K chars/read).
2. Write the analysis file in the FIRST turn after reading — a full draft
   (cluster map, first-extraction recommendation, seam/test plan, risks,
   verdict) from what you have. Target >10KB; verify size after writing.
3. Then deepen: run the greps below and patch the file with cited evidence
   (exact line numbers, counts, caller examples). Do not pre-empt the write
   with long investigation — the file must exist first.

## Evidence greps (all read-only, repeatable)
- Def inventory + spans:
      grep -nE "^(def |class |async def )" <file> | awk -F: '$1 >= LO && $1 <= HI'
  Find each def's end from the NEXT def line; flag defs that straddle the
  region edges (def line < region start, or body past region end) — they must
  be carried wholesale by extraction, never line-sliced.
- Helper / constant locations for the dependency table:
      grep -n "def <sym>" <file> ; grep -n "^<CONST> *=" <file>
  Cite exact lines. Note FORWARD deps into later regions (e.g. a default
  constant defined in R4 consumed by an R3 function) — extraction must import
  it, not re-define it.
- External callers (re-export seam proof):
      grep -rn "<alias>\.<fn>\b" <pkg>/ | grep -v "<godfile>"
      grep -rln "import <godfile>|from .*<godfile> import" <pkg>/
  Callers using a module alias (`from pkg import godfile as kb; kb.claim_task`)
  mean extraction MUST keep host-module re-exports (`from .sub import fn`)
  or every call site breaks. The `kb.<fn>` counts also identify which
  functions are only reachable from the dispatcher/tests (0 external refs).
- Test coverage per function:
      grep -rn "<fn>" tests/ | wc -l
  Functions with 0 refs are coverage gaps — name them explicitly in the test
  plan and risks; they are the most likely silent-regression spots.

## Blind discipline
Wave-1 analysis files for other passes live in the SAME campaign directory as
your output (e.g. `R3-analysis.md` sits next to `R3-analysis-B.md`). Never
open them. Name your output `<region>-analysis-B.md` and write ONLY that one
file. Confirm read-only at the end via the blob-hash check above.
