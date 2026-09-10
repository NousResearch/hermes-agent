# Skill Verification Process

A repeatable procedure for installing the bundled skills into a profile and
confirming the development skills are functional. Run from the repo root with
the Hermes CLI (`hermes`, aliased to `gizmo` in some setups).

## 1. Install / re-seed bundled skills

Bundled skills ship under `skills/` and are seeded into the active profile's
`~/.hermes/skills/` directory. If `hermes skills list` shows `0 builtin`, the
profile has not been seeded (or was opted out). Re-seed immediately with:

```bash
hermes skills opt-in --sync
```

This removes the `.no-bundled-skills` marker and copies every bundled skill into
the profile. Expected output: `Re-seeded 72 bundled skill(s).`

## 2. Verify registration (local builtins)

Confirm the skills registered and are enabled. Use `list`, not `inspect`:

```bash
hermes skills list --source builtin --enabled-only
```

All 9 `software-development` skills (`hermes-agent-skill-authoring`, `plan`,
`systematic-debugging`, `test-driven-development`, `requesting-code-review`,
`simplify-code`, `spike`, `node-inspect-debugger`, `python-debugpy`) should
appear as `builtin / enabled`.

The platform-gated skills (e.g. the macOS-only `apple/*` set) are correctly
hidden on Linux, so the enabled count is lower than the seeded count (65 of 72
on Linux).

## 3. Run a functional test task

Exercise a skill end to end. Example using `test-driven-development` in a
throwaway workspace, following its RED -> GREEN cycle with the project test
runner (`pytest`):

```bash
mkdir -p /tmp/tdd_verify && cd /tmp/tdd_verify
# RED: write the test against a stub, watch it fail for the right reason
# GREEN: write minimal code, watch it pass
/home/vkj/.hermes/hermes-agent/venv/bin/python -m pytest test_retry.py -q
```

A correct run shows the test failing with a real assertion error (feature
missing, not a typo/import error) in RED, then `1 passed` in GREEN. Remove the
temp workspace afterward.

## Caveats and resolutions

- **`hermes skills inspect <name>` resolves to remote registries.** `inspect`
  only takes a registry identifier (or an `http(s)` SKILL.md URL) and rejects
  local paths, so a bare name matches community skills on skills.sh rather than
  the installed builtin. To inspect a local builtin, use
  `hermes skills list --source builtin` and read the installed
  `~/.hermes/skills/<category>/<skill>/SKILL.md`.

- **`pytest` was missing from the project `venv`.** The dev test tooling is
  pinned in the `dev` extra of `pyproject.toml`. Install it into the venv so the
  skills' documented `pytest` commands work:

  ```bash
  venv/bin/python -m pip install "pytest==9.0.2" "pytest-asyncio==1.3.0"
  ```

  For full CI parity, prefer `scripts/run_tests.sh` over calling `pytest`
  directly.
