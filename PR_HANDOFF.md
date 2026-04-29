## Self-Correction Guard — PR Handoff

**Status:** Patch applied and tested. Ready for PR.

### What was done
- Added `SAFER_ALTERNATIVES` dict + `_get_safer_hint()` to `tools/approval.py`
- Injected hints into 4 code paths (2 in `check_dangerous_command`, 2 in `check_all_command_guards`)
- Added `"hint"` field to approval data sent to gateway notification callback
- Added `tests/tools/test_self_correction.py` (9 tests)
- All 128 tests pass (119 existing + 9 new)

### Files changed
- `tools/approval.py` — new `SAFER_ALTERNATIVES` dict, `_get_safer_hint()` fn, modified 4 code paths
- `tests/tools/test_self_correction.py` — new test file

### How it works
When a dangerous command is detected by pattern matching, the model now receives:
```
⚠️ script execution via -e/-c flag. Asking the user for approval.
Hint: try a safer alternative. Use the execute_code tool, or write the script to a .py file first.
```
This mirrors Tirith's existing hint mechanism but for pattern-based detection.

### Patterns with hints (19 total)
script execution via -e/-c flag, shell command via -c/-lc flag, script execution via heredoc,
pipe remote content to shell, execute remote script via process substitution, recursive delete,
delete in root path, overwrite system config, overwrite system file via tee, overwrite system file
via redirection, in-place edit of system config, stop/disable system service, git reset --hard,
git force push (both variants), git clean with force, git branch force delete,
world/other-writable permissions (both variants)

### Patterns WITHOUT hints (truly destructive — just block)
fork bomb, format filesystem, disk copy, write to block device, SQL DROP/TRUNCATE/DELETE,
kill all processes, force kill, kill hermes process (self-termination)

### To create PR
1. `cd /home/ubuntu/.hermes/hermes-agent`
2. `git checkout -b feat/self-correction-guard`
3. `git add tools/approval.py tests/tools/test_self_correction.py`
4. `git commit -m "feat(approval): inject safer-alternative hints before user escalation"`
5. `git push origin feat/self-correction-guard`
6. `gh pr create --title "feat: self-correction guard — try safer alternatives before user escalation" --body-file PR_HANDOFF.md`
