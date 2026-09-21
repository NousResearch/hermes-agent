---
name: hermes-profile-handoff
description: Mandatory protocol when switching between Hermes profiles (Kensei, Moss, Upstream) or before any git commit/branch/push. Prevents wrong-repo commits, pip install corruption, and CLI symlink rot.
triggers:
  - profile handoff
  - switch from moss to kensei
  - switch from kensei to moss
  - which repo am I in
  - wrong profile
  - hermes CLI looks vanilla
  - prevent wrong-repo commits
  - recover after profile mismatch
---

# Hermes Profile Handoff Protocol

## Why this exists

The VPS runs two parallel Hermes installations that share system Python:

| Profile | Repo | Venv |
|---|---|---|
| `default` (Kensei) | `~/repos/KenseiAgent` (custom fork, runtime) | `KenseiAgent/.venv` |
| `moss`/`upstream` | `~/repos/hermes-agent-vanilla` (vanilla upstream, PRs) | `hermes-agent-vanilla/.venv` |

When an agent switches from one to the other without cleaning up, the state corrupts:
- `pip show hermes-agent` points to the wrong repo
- `~/.local/bin/hermes` gets a bad shebang
- Git commits land in the wrong repo
- `pydantic_core._pydantic_core` raises ModuleNotFoundError (wrong arch wheel cached)

## Profile detection

The active profile is read from two sources (env var takes precedence):

```bash
# Per-session override
HERMES_PROFILE=moss hermes --version
HERMES_PROFILE=default hermes --version

# Persistent (file)
cat /home/kensei/.hermes/profile
```

If neither is set, the dispatcher defaults to `default` (KenseiAgent).

## The 30-second handoff checklist

Run this EVERY time you finish a session and are about to hand control to another agent:

```bash
# 1. Are you on the right repo?
cd ~/repos/KenseiAgent && git rev-parse --show-toplevel
# Expected: /home/kensei/repos/KenseiAgent

# 2. Are you on the right branch?
cd ~/repos/KenseiAgent && git branch --show-current
# Expected: main (or your active feat/ branch)

# 3. Both repos clean?
cd ~/repos/KenseiAgent && git status --short
cd ~/repos/hermes-agent-vanilla && git status --short
# Both should be clean (or have expected work)

# 4. Profile matches repo?
cat /home/kensei/.hermes/profile
# If working on KenseiAgent, should say "default"
# If working on upstream, should say "moss"
```

## The CLI dispatcher

`/home/kensei/.local/bin/hermes` is a **bash script** (not a symlink, not pip-managed). It routes by profile. **Never replace it with a pip entry point or symlink** — those get overwritten.

If missing, restore:

```bash
cat > /home/kensei/.local/bin/hermes << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
declare -A M=(
    [default]="/home/kensei/repos/KenseiAgent/.venv/bin/hermes"
    [kensei]="/home/kensei/repos/KenseiAgent/.venv/bin/hermes"
    [moss]="/home/kensei/repos/hermes-agent-vanilla/.venv/bin/hermes"
    [upstream]="/home/kensei/repos/hermes-agent-vanilla/.venv/bin/hermes"
)
P="${HERMES_PROFILE:-}"
[[ -z "$P" && -f /home/kensei/.hermes/profile ]] && P=$(cat /home/kensei/.hermes/profile)
[[ -z "$P" ]] && P="default"
T="${M[$P]:-}"
[[ -z "$T" || ! -x "$T" ]] && { echo "error: profile '$P' venv not found" >&2; exit 1; }
exec "$T" "$@"
EOF
chmod +x /home/kensei/.local/bin/hermes
```

## Pre-commit hooks

Both repos have `core.hooksPath=.githooks` configured so pre-commit hooks block wrong-repo commits:

| Repo | Blocks when profile is | Allows when profile is |
|---|---|---|
| KenseiAgent | `moss`, `upstream` | `default`, `kensei`, unset |
| upstream | `default`, `kensei`, unset | `moss`, `upstream` |

Override: `git commit --no-verify`

## Branch safety (critical)

Before ANY `git commit`, `git checkout -b`, or `git push`:

1. `git rev-parse --show-toplevel` — confirms the repo
2. `git branch --show-current` — confirms the branch
3. `git status --short` — confirms the right work is tracked

For upstream work (Mossy): branch from `upstream/main`:
```bash
cd ~/repos/hermes-agent-vanilla
git fetch upstream
git checkout -b fix/issue-XXXX upstream/main
```

For Kensei work: branch from local `main`:
```bash
cd ~/repos/KenseiAgent
git checkout main && git pull
git checkout -b feat/my-feature
```

## Recovery from broken state

If `hermes --version` shows vanilla / wrong project:

```bash
# 1. Fix profile
echo "default" > /home/kensei/.hermes/profile

# 2. Check dispatcher exists
ls -la /home/kensei/.local/bin/hermes  # should be a bash script, not symlink

# 3. Fix pip install if editable location is wrong
pip install --break-system-packages -e /home/kensei/repos/KenseiAgent 2>&1 | tail -3

# 4. Fix pydantic_core if cross-arch issue
pip install --force-reinstall pydantic-core --break-system-packages 2>&1 | tail -3
python3 -c "import pydantic_core._pydantic_core; print('OK')"

# 5. Verify
hermes --version
# Expected: Hermes Agent v0.17.0 ... upstream ad5867c7 · local ... (+N carried commits)
```
