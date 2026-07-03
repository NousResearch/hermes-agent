"""Provisioning: create and lock down a per-employee HERMES_HOME.

Each employee gets an isolated Hermes data dir containing their own config,
secrets, sessions, skills, memory and cron — pointed at the internal inference
endpoint. Nothing here is shared with another employee.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import Settings
from .models import Employee
from . import security

_FETCH_SRC = Path(__file__).parent / "worker_bin" / "orchard-fetch"

# Standard Hermes subdirs we pre-create so first run is fast and perms are right.
_SUBDIRS = ["skills", "sessions", "memories", "cron", "logs", "workspace", "home", "run"]
_SECRET_FILES = {".env", "auth.json"}


def scaffold_home(settings: Settings, employee: Employee) -> Path:
    """Create + harden the employee's HERMES_HOME. Idempotent."""
    home = settings.paths.home_for(employee.id)
    security.assert_contained(home, settings.paths.employees_dir())

    sec = settings.security
    security.ensure_dir(home, sec.home_mode_int)
    for sub in _SUBDIRS:
        security.ensure_dir(home / sub, sec.home_mode_int)

    write_hermes_config(settings, home)
    _write_worker_secrets(settings, home)

    # Enforce perms across the whole tree, then optionally hand the tree to a
    # dedicated unprivileged OS user (the real per-worker boundary in prod).
    security.harden_tree(home, sec.home_mode_int, _SECRET_FILES, sec.secret_mode_int)
    if sec.run_as_user:
        security.chown_recursive(home, sec.run_as_user)
    return home


def write_hermes_config(settings: Settings, home: Path) -> None:
    llm = settings.llm
    # Hermes reads config.yaml at HERMES_HOME root. model = "provider:model".
    config = (
        f"model: {llm.provider}:{llm.model}\n"
        f"agent:\n"
        f"  reasoning_effort: {llm.reasoning_effort}\n"
        f"terminal:\n"
        f"  backend: local        # the sandbox IS the boundary (see backends/)\n"
        f"  working_dir: workspace\n"
    )
    disabled = settings.skills.disabled or []
    if disabled:
        # Disable competing bundled skills (e.g. github-auth) so the agent uses
        # our custom ones and doesn't wander off into `gh auth login` etc.
        config += "skills:\n  disabled:\n" + "".join(f"    - {d}\n" for d in disabled)
    # Disable arbitrary code execution: it always needs approval AND is a
    # security hole (bypasses terminal restrictions). Forces the agent through
    # the terminal, where orchard-fetch / plain curl work without a prompt.
    config += "disabled_toolsets:\n  - code_execution\n"
    (home / "config.yaml").write_text(config)


def _write_worker_secrets(settings: Settings, home: Path) -> None:
    llm = settings.llm
    prefix = llm.provider.upper()
    lines = [
        f"{prefix}_BASE_URL={llm.base_url}",
        f"{prefix}_API_KEY={llm.api_key}",
    ]
    security.write_secret(
        home / ".env", "\n".join(lines) + "\n", settings.security.secret_mode_int
    )


_SOUL = """# Operating rules (always follow)

You are a work assistant for one employee. Some requests need an external
integration (GitHub, Jira, Wiki). Each integration's token is provided to your
tools as an environment variable (e.g. `GITHUB_TOKEN`, `JIRA_TOKEN`).

## If an integration's token env var is EMPTY
Do EXACTLY this, and nothing else:
1. Mint a secure one-time entry link (replace <id> with the integration id,
   e.g. `github`, `jira`, `wiki`):
   `curl -s -X POST "$ORCHARD_API/api/employees/$ORCHARD_EMPLOYEE_ID/integrations/<id>/link"`
   It returns `{"url": "..."}`.
2. Reply with that URL and ask the user to open it and paste the token there.
3. STOP. NEVER ask the user to type or paste a token into this chat. NEVER
   install a CLI (like `gh`), use an unauthenticated/public API, or search the
   filesystem to work around a missing token.

When the token env var IS set, use it directly (the matching skill has the calls).

## Calling integration APIs
Use the pre-approved helper `"$HERMES_HOME/bin/orchard-fetch" "<url>"` — it only
reaches allowlisted integration domains, injects the token itself, and does NOT
prompt for approval. Do NOT use `curl`/`gh` or pipe anything into an interpreter
(`python3`, `jq`, `sh`) — that trips the security scanner. Read the JSON it prints
and summarize it yourself.
"""


def write_soul(settings: Settings, employee_id: str) -> None:
    """Write the always-in-context operating rules (SOUL.md) for a tenant."""
    (settings.paths.home_for(employee_id) / "SOUL.md").write_text(_SOUL)


def sync_shared_skills(settings: Settings, employee_id: str) -> int:
    """Copy the admin-curated base skills into the tenant's own skills dir so the
    worker's Hermes actually discovers them (Hermes only scans HERMES_HOME/skills,
    and the sandbox blocks the shared dir under /Users anyway). Returns count."""
    shared = settings.skills.shared_dir
    if not shared or not Path(shared).is_dir():
        return 0
    dest = settings.paths.home_for(employee_id) / "skills"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for child in Path(shared).iterdir():
        if child.is_dir():
            shutil.copytree(child, dest / child.name, dirs_exist_ok=True)
            n += 1
    return n


def install_fetch_helper(settings: Settings, employee_id: str) -> None:
    """Install the pre-approved `orchard-fetch` helper + the domain allowlist
    (integrations.json) into the tenant home, so the agent can reach integration
    URLs without an approval prompt, only for allowlisted domains."""
    from . import integrations
    home = settings.paths.home_for(employee_id)
    bindir = home / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    dst = bindir / "orchard-fetch"
    dst.write_text(_FETCH_SRC.read_text())
    dst.chmod(0o755)
    (home / "integrations.json").write_text(
        json.dumps(integrations.fetch_allowlist(settings), indent=2))


def deprovision(settings: Settings, employee_id: str) -> bool:
    """Remove an employee's home entirely. Returns True if something was deleted."""
    home = settings.paths.home_for(employee_id)
    security.assert_contained(home, settings.paths.employees_dir())
    if home.exists():
        shutil.rmtree(home)
        return True
    return False
