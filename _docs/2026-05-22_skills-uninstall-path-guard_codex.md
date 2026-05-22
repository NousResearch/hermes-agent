# Skills uninstall path guard

Date: 2026-05-22

## Scope

- Hardened `tools/skills_hub.py` so lock-file `install_path` values are validated when saved and when used for uninstall deletion.
- Kept the change limited to Skills Hub path handling and focused tests.

## Why

`uninstall_skill()` used the lock-file `install_path` directly under `SKILLS_DIR`. A corrupted or stale lock entry could point at an absolute path, a traversal path, or the skills root itself. The new guard accepts only `skill` or `category/skill`, requires the final segment to match the requested skill name, resolves the path, and refuses targets outside `SKILLS_DIR`.

## Verification

```powershell
py -3.12 -X utf8 -B -m pytest -o addopts= tests/tools/test_skills_hub.py::TestHubLockFile tests/tools/test_skills_hub.py::TestUninstallSkillPathGuard -q -p no:cacheprovider -p no:randomly
py -3.12 -X utf8 -B -m compileall tools/skills_hub.py
py -3.12 -X utf8 -B -m ruff check tools/skills_hub.py tests/tools/test_skills_hub.py
```

Focused result: `17 passed, 1 skipped` (directory symlink privilege unavailable on this Windows host).

Note: a full `tests/tools/test_skills_hub.py` run was attempted, but unrelated existing tests failed in bundle hash / optional-skill binary asset areas, and one temp-directory setup error occurred on this Windows host.
