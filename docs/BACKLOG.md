# Backlog

## Open

_None._

## Completed

- **Source:** Windows canonical skill-tool suite · **Severity:** medium / cross-platform API contract · **Description:** on Windows, the `skill_view` JSON `path` and collision `matches` fields used native `\\` separators even though categorized skill identifiers and the tested tool contract use `/`. · **Status:** DONE — all user/model-facing skill paths now use `Path.as_posix()` normalization; the complete `test_skills_tool.py` wrapper gate verifies the behavior.
- **Source:** Windows canonical test-wrapper gate · **Severity:** medium / developer infrastructure · **Description:** when invoked from MSYS, `scripts/run_tests.sh` recognized only the POSIX `venv/bin/python` layout, then `env -i` discarded the home, system, temp, and UTF-8 variables required by native Windows Python. As a result, the mandatory wrapper could not find the venv, failed before collection with `Path.home()`/CP1250 errors, or created a relative `%SystemDrive%/ProgramData` cache tree in the worktree. · **Status:** DONE — the wrapper recognizes the `Scripts/python.exe` layout, forwards the required non-secret Windows environment variables, and runs with `PYTHONUTF8=1`; verified through the real wrapper and a clean-worktree readback.
- **Source:** Curator write-surface pytest gate · **Severity:** low / test portability · **Description:** `TestDeleteSkillRmtreeGuard.test_symlinked_skill_dir_refused` failed with `WinError 1314` before reaching the guard under test when the Windows process lacked symlink privileges. · **Status:** DONE — following the repository's existing capability-gate pattern, the test runs only in symlink-capable environments; where supported, the original refusal and external-target integrity assertions remain unchanged.
