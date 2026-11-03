# Python compatibility audit and migration plan

**Status:** audit only; this document does not claim that Python 3.13 or 3.15 is supported.

## Findings from the current repository configuration

- `.python-version` selects Python `3.14`.
- `pyproject.toml` declares `requires-python = ">=3.11,<3.15"`, but its comments explicitly say that the application supports only Python 3.14 and allows older Python versions so existing installations can pass through the updater.
- The main runtime dependency declarations are guarded by markers such as `python_version >= '3.14'`. Therefore, the broad `requires-python` range is not proof that the application can run on Python 3.11–3.13.
- The dependency lock and the managed-Python setup are part of the compatibility contract. Changing only `requires-python`, `.python-version`, or a CI matrix would be unsafe and could create installs that start updating but fail at runtime.
- This audit has not run the full test suite under Python 3.13, 3.14, or a development build of 3.15. No compatibility claim should be made until CI and platform-specific tests pass.

## Proposed compatibility policy

1. Keep Python 3.14 as the known supported runtime until a complete alternate dependency resolution exists.
2. Treat Python 3.13 as a compatibility target to investigate, not as supported yet. Resolve every direct and transitive dependency for 3.13, including optional extras and platform-specific packages, then generate and validate a lock for that interpreter.
3. Treat Python 3.15 as a forward-compatibility target only when a usable interpreter and compatible wheels/dependencies are available. Do not promise support based solely on syntax or metadata.
4. Preserve the updater bridge for older existing installations. Do not conflate “the updater can run” with “the full agent is supported.”
5. Add a CI matrix only after dependency resolution works. Run import/smoke tests, core tests, provider-extra tests, and platform-specific checks for each claimed Python version.
6. Test Android/Termux separately: Android platform markers, psutil's Android-specific source pin, and wheel availability differ from desktop Linux.

## Required implementation sequence

- Inventory Python-version markers and all dependency groups/extras in `pyproject.toml`.
- Resolve dependencies separately for each target Python version and supported OS/architecture.
- Identify syntax/API incompatibilities and packages without compatible wheels.
- Add version-specific lock/resolution handling without weakening exact pins or supply-chain controls.
- Add CI jobs and a small runtime smoke suite for each version.
- Run the full relevant tests and document any exclusions before advertising support.

## Acceptance criteria

- Each supported version has a reproducible dependency resolution.
- Fresh install, updater, CLI startup, gateway startup, and representative tool/provider paths pass.
- CI results are green for the declared OS/architecture matrix.
- Documentation distinguishes the updater compatibility range from supported application runtimes.

## Current conclusion

The repository configuration currently selects Python 3.14 as the supported runtime. Python 3.13 compatibility and Python 3.15 forward compatibility remain unverified. This PR records the evidence and a safe migration plan; it does not change runtime pins or claim the compatibility work is complete.
