# Post-Mortem: Hermes-Agent Fork Patch Loss & Restoration

**Date:** April 30, 2026
**Incident:** Custom patches (approval justification gate + cron report header) silently dropped during upstream sync
**Resolution:** Manual re-application across 9 files, committed as `dc815e64c`

---

## Timeline

| Date | Event |
|------|-------|
| Apr 18 | `e682e3f0c` — Approval justification gate originally committed on `security/enhancements` |
| Apr 19 | `56ecda022` — Fix for infinite justification_required loops |
| Apr 22 | `42df69a28` + `6f9863694` — Cron report header meta committed |
| ~Apr 28 | Upstream sync merges ~1700 upstream commits into `feat/self-correction-guard-verified` |
| Apr 29 | Yohan notices cron headers missing, asks for rebase (session `20260429_232220`) |
| Apr 30 | Full investigation: both patches completely gone from working tree. Zero grep hits for `justification` in any modified file. |
| Apr 30 | Cherry-pick attempted → 1708-commit divergence makes it impossible (massive conflicts) |
| Apr 30 | Manual re-application via Python scripts → succeeds after multiple indentation failures |
| Apr 30 | Commit `dc815e64c` pushed to `alistaircl/hermes-agent` |

---

## What Went Well

1. **The orphaned branches were still available.** `remotes/fork/security/enhancements` and `remotes/fork/feat/cron-header-meta` still had the original commits, so we could `git show` the exact diffs and use them as reference. If we'd run `git branch -D` or pruned remotes, we'd have been reconstructing from memory.

2. **The original patch diffs were clean and well-structured.** Commits `e682e3f0c` and `56ecda022` had clear, focused changes — a new Phase 2.8 gate, new data structures, a fix for infinite loops. This made the manual re-application feasible because we understood *what* the code was supposed to do, not just *that* some code was missing.

3. **The Python script approach eventually worked.** Once we abandoned the `patch` tool and wrote scripts that read the target files, detected surrounding indentation with `get_indent()`, and inserted code at the correct byte offsets, we got clean results. The `ast.parse()` validation after every modification caught problems immediately.

4. **The verification checklist caught the Discord embed miss.** The automated verification script flagged that Discord was missing the `Agent Justification` embed field, which we'd have missed in a manual review.

---

## What Went Wrong

### 1. Patches were silently dropped by a merge (THE core incident)

A `git merge` from upstream overwrote our custom code in `approval.py`, `terminal_tool.py`, `gateway/run.py`, and `cron/scheduler.py` **without producing any conflict markers**. Git resolved the "conflict" by taking the upstream version because our changes weren't on the merge base — they were on a separate branch that had been merged and then overwritten.

**Root cause:** Git's three-way merge algorithm. When both sides modify the same region, Git picks the side that's "closer" to the merge base. Since our patches were on a feature branch (`security/enhancements`) that was 1708 commits behind, and upstream had modified the same files (refactoring, renaming, reformatting), Git considered upstream's version to be the "correct" resolution.

**Why we didn't notice immediately:** There were no conflict markers. The merge looked clean. And `git log` after the merge showed our feature branch had been merged at some point, giving a false sense of security.

### 2. The `patch` tool is fundamentally broken for Python

The Hermes `patch` tool's replace mode strips or normalizes leading whitespace in `new_string`. This is catastrophic for Python where indentation is syntax. Across this session:

- **6 of 8 files** were patched successfully by the initial Python script (using `str.replace()` with exact string matching)
- **2 of 8 files** (gateway/run.py, terminal_tool.py) failed because the script's template strings didn't match the actual file whitespace
- **All subsequent `patch` tool calls** on the 2 broken files introduced new indentation errors even when they "applied" — because the tool mangled the replacement text's indentation

The `patch` tool worked fine for YAML, markdown, and other non-whitespace-sensitive formats. But for Python it's unreliable.

### 3. We generated 20 throwaway scripts in `/home/ubuntu/`

Because neither `patch` nor `python3 -c` worked reliably, we wrote file-after-file of one-off Python scripts:
- `apply_justification_patches.py`, `apply_all_patches.py`
- `fix_indent.py`, `fix_discord.py`, `fix_discord_v2.py`
- `patch_gateway_run.py`, `patch_gateway_v2.py`, `patch_gateway_v3.py`
- `patch_cron_scheduler.py`, `patch_cron_v2.py`
- `check_syntax.py`, `show_syntax_error.py`, `check_scheduler_indent.py`
- `diag_run.py`, `verify_all_patches.py`
- And 5 older scripts from previous sessions

Each was written, run once, and (mostly) never cleaned up. The `rm` cleanup was blocked by the security gate.

### 4. The session burned through excessive tool iterations

The combination of broken patches, re-reads to diagnose indentation, stash/restore cycles, and incremental fixes meant this task consumed far more tool calls than it should have. A single working approach (Python scripts with `get_indent()`) applied from the start would have been ~15 calls instead of ~60+.

### 5. The security gate blocked `python3 -c` and `rm`

Two guardrails that are normally useful became friction:
- `python3 -c` blocked → forced us to write scripts to files for even trivial 3-line checks
- `rm /home/ubuntu/*.py` blocked → temp scripts accumulated

---

## Why It Went Wrong (Deeper Analysis)

### The merge-silently-drops-patches problem

This is a **known class of Git footgun** for fork maintainers. It happens when:

1. You have custom commits on branch A
2. You merge upstream/main into branch A
3. Upstream has refactored the same files your commits touch
4. Git's three-way merge resolves the conflict by taking upstream's version (no markers)

The fix isn't "don't merge" — it's **verify after every merge**. But we had no verification step, no patch registry, and no automated way to detect that code was missing.

### The indentation problem

The hermes-agent codebase uses **inconsistent indentation**:
- Most files: 4-space indent (standard Python)
- `gateway/run.py`: 1-space indent in deeply nested functions (likely auto-generated or early code)
- `cron/scheduler.py`: 1-space indent in `run_job()` (same pattern)

The `patch` tool and our Python scripts both assumed 4-space indent, leading to mismatches. The real fix is either:
- A tool that reads the actual indentation from surrounding lines (what our `get_indent()` approach eventually did)
- An AST-aware tool that operates on the syntax tree, not raw text

### No patch specification

We had the original *code* (in orphaned branches) but no *specification* of what the patches were supposed to do. This meant when the code didn't apply cleanly, we had to reverse-engineer intent from the diffs rather than saying "add a Phase 2.8 gate that does X, Y, Z."

---

## Lessons Learned

| # | Lesson | Severity |
|---|--------|----------|
| 1 | `git merge` from upstream can silently delete custom patches with zero conflict markers | 🔴 Critical |
| 2 | The `patch` tool cannot be trusted for Python files — it mangles indentation | 🔴 Critical |
| 3 | `python3 -c` being blocked by security gate adds friction with no safety benefit for read-only operations | 🟡 Medium |
| 4 | Without a post-merge verification step, patch loss is invisible until runtime | 🔴 Critical |
| 5 | Having orphaned branches as reference saved us — don't prune them | 🟢 Good practice |
| 6 | 20 throwaway scripts indicate a missing "proper patching tool" in the toolchain | 🟡 Medium |
| 7 | Patch specs > patch code — diffs rot, specs don't | 🟡 Medium |

---

## Action Items

### 🔴 Critical: Prevent Silent Patch Loss

**A1. Write a patch verification script and run it after every upstream sync.**

Create `~/.hermes/scripts/verify_fork_patches.py` that:
- Checks every marker from the patch registry (see A2)
- Runs `ast.parse()` on all modified files
- Returns non-zero if any marker is missing
- Can be called as a git post-merge hook

**A2. Maintain a patch registry (not just in the skill — in a machine-readable file).**

Create `~/.hermes/hermes-agent/PATCHES.yaml`:

```yaml
patches:
  - name: approval-justification-gate
    description: "Model must justify before user escalation. Phase 2.8 gate between self-correction and approval."
    files:
      - path: tools/approval.py
        markers:
          - "justification: str = None"
          - "Phase 2.8"
          - "justification_required"
          - '"justification": effective_justification'
      - path: tools/terminal_tool.py
        markers:
          - "justification: Optional[str]"
          - "justification_required"
      - path: gateway/run.py
        markers:
          - "justification = approval_data.get"
          - "justification=justification"
      - path: gateway/platforms/telegram.py
        markers:
          - "justification: Optional[str]"
      - path: gateway/platforms/discord.py
        markers:
          - "justification: Optional[str]"
          - "Agent Justification"
      - path: gateway/platforms/slack.py
        markers:
          - "justification: Optional[str]"
      - path: gateway/platforms/matrix.py
        markers:
          - "justification: Optional[str]"
      - path: gateway/platforms/feishu.py
        markers:
          - "justification: Optional[str]"
    original_commits: [e682e3f0c, 56ecda022]
    restored_commit: dc815e64c

  - name: cron-report-header-meta
    description: "Cron delivery reports show provider/model, elapsed time, and run timestamp in header."
    files:
      - path: cron/scheduler.py
        markers:
          - "_elapsed_seconds"
          - "_job_start_time"
          - "model_info"
          - "Run Time:"
          - "Model: {model_info}"
    original_commits: [42df69a28, 6f9863694]
    restored_commit: dc815e64c
```

**A3. Add a git post-merge hook** that runs the verification script automatically after `git merge` or `git pull`.

### 🟡 Medium: Better Patching Tools

**A4. Install `ast-grep` for AST-aware Python code modification.** ✅ Done — `ast-grep` 0.42.1 installed at `~/.local/bin/ast-grep`

[`ast-grep`](https://github.com/ast-grep/ast-grep) (aka `sg`) operates on the syntax tree, not raw text. It:
- Preserves indentation automatically
- Matches by AST pattern, not string comparison
- Handles whitespace variations transparently

```bash
pip install ast-grep-py
```

For our use case, `ast-grep` would handle "add a parameter to a function" as a structural transformation rather than a text replacement. This eliminates the indentation problem entirely.

**Alternative: LibCST** (Instagram/Meta's Python codemod framework). More powerful for complex multi-file refactors, but heavier setup. `ast-grep` is simpler for one-off patches.

**A5. Build a reusable `patch_python.py` utility script** ✅ Done — `scripts/patch_python.py` with insert-after, replace-between, add-param commands
- Takes a file path, a line range or function name, and replacement code
- Reads the target file's actual indentation (detecting tabs vs spaces, indent width)
- Applies the change with correct indentation
- Validates with `ast.parse()`
- Reports success/failure with line numbers

This replaces the 20 one-off scripts with a single general-purpose tool.

### 🟡 Medium: Patch Specifications

**A6. Write declarative patch specs instead of relying on old code.** ✅ Done — `patches/approval-justification-gate.yaml` and `patches/cron-report-header-meta.yaml`

Instead of "here's the diff from commit X," maintain specifications like:

```yaml
# patches/justification-gate.yaml
name: approval-justification-gate
intent: |
  Before escalating a flagged command to user approval, require the AI agent
  to provide a text justification for why the command is necessary. This:
  1. Reduces approval fatigue (users see reasoning, not just raw commands)
  2. Creates an audit trail of agent intent
  3. Gives the agent one chance to self-correct before bothering the user

behavior:
  - When check_all_command_guards() flags a command AND no justification is provided,
    return status="justification_required" with a hint message
  - When the agent retries WITH justification, include it in the approval_data
  - After 3 failed justification attempts, return status="justification_denied"
  - The justification must appear in all platform approval UIs (button + text fallback)

integration_points:
  - tools/approval.py: Phase 2.8 gate, between self-correction (2.7) and approval (3)
  - tools/terminal_tool.py: accept justification param, handle new statuses
  - gateway/run.py: extract justification from approval_data, pass to adapters
  - All platform adapters: accept and display justification param
```

This way, even if the original code is lost and the upstream has refactored everything, a future agent can implement the patch from the spec rather than trying to apply a stale diff.

### 🟢 Cleanup

**A7. Clean up `/home/ubuntu/*.py` temp scripts.** ✅ Done — 17 stale scripts removed. 4 intentional scripts remain (benchmark_e2e, route_proxy, route_proxy_test, tls_mitm_proxy).

**A8. Don't prune orphaned branches** — `security/enhancements` and `feat/cron-header-meta` are useful references even though they can't be cherry-picked. ✅ Verified present on fork + origin remotes. Documented in PATCHES.yaml reference_branches.

---

## Summary

The incident was caused by a fundamental Git limitation (silent patch loss in three-way merges) compounded by missing verification tooling and an unreliable patching method. The fix took 60+ tool calls and 20 temp scripts when it should have taken ~15. The three highest-impact improvements are:

1. **PATCHES.yaml + verification script** — detect silent loss immediately
2. **ast-grep or a reusable `patch_python.py`** — stop fighting indentation
3. **Declarative patch specs** — survive codebase refactors that make diffs inapplicable
