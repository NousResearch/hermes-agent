"""Boot-time provisioning of per-profile git credentials.

Make "the profile's own GitHub token usable by git" happen in the agent
runtime at boot, rather than relying on an external management layer to
reach into the runtime's data directory from the outside:

  * The tool subprocess (where the agent runs ``git``) deliberately CANNOT
    see ``GITHUB_TOKEN`` — provider secrets are blocklisted from the
    subprocess env. So a credential file is the only path, and a skill that
    runs *inside* the subprocess can't read the raw token to write one.
  * This module runs in-process at container boot, where the profile's
    ``.env`` is readable on the mounted volume and the subprocess HOME
    (``{HERMES_HOME}/home``, per ``get_subprocess_home()``) is known
    directly.

It writes the profile's OWN token (from its ``.env``) into
``{HERMES_HOME}/home/.git-credentials`` + ``.gitconfig`` so git's ``store``
helper picks it up. Tokens never cross-pollinate — each home reads only its
own ``.env``. Creating ``{HERMES_HOME}/home`` is also what activates the
tool subprocess's HOME override (see ``get_subprocess_home``).

Provides a pure, per-home function plus a profile-aware ``provision_all``
and a ``main()`` entry point intended to run during the runtime boot
sequence. The runtime owns the credential path, so no external component
needs to know about it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

GIT_HOST = "github.com"

# Checked in order; a dedicated GITHUB_PAT wins over the copilot GITHUB_TOKEN.
TOKEN_VARS = ("GITHUB_PAT", "GITHUB_TOKEN", "GH_TOKEN")


# ---------------------------------------------------------------------------
# Pure content builders
# ---------------------------------------------------------------------------


def build_git_credentials_content(token: str) -> str:
    """The ``~/.git-credentials`` line the ``store`` helper reads for HTTPS."""
    return f"https://x-access-token:{token}@{GIT_HOST}\n"


def build_git_config_content(*, name: str, email: str) -> str:
    """Minimal ``~/.gitconfig``: store helper + identity + ssh→https rewrites."""
    return "\n".join(
        [
            "[credential]",
            "\thelper = store",
            "[user]",
            f"\tname = {name}",
            f"\temail = {email}",
            f'[url "https://{GIT_HOST}/"]',
            f"\tinsteadOf = git@{GIT_HOST}:",
            f"\tinsteadOf = ssh://git@{GIT_HOST}/",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# .env token extraction
# ---------------------------------------------------------------------------


def _read_env_token(env_path: Path) -> Optional[tuple[str, str]]:
    """Return ``(token, source_var)`` from ``env_path`` or ``None``.

    Reads a flat ``.env`` file (no shell expansion). The first non-empty
    match across ``TOKEN_VARS`` (in precedence order) wins.
    """
    try:
        content = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for key in TOKEN_VARS:
        m = re.search(rf"^{re.escape(key)}=(.*)$", content, re.MULTILINE)
        if m:
            val = re.sub(r'^["\']|["\']$', "", m.group(1).strip())
            if val:
                return val, key
    return None


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvisionResult:
    home: Path
    provisioned: bool
    reason: Optional[str] = None
    source: Optional[str] = None


def provision_git_credentials(
    home_dir: Path,
    *,
    name: Optional[str] = None,
    email: Optional[str] = None,
    force: bool = False,
) -> ProvisionResult:
    """Provision git auth for one profile home from its own ``.env`` token.

    ``home_dir`` is a HERMES_HOME-shaped directory: it holds ``.env`` at the
    root and the tool subprocess HOME at ``{home_dir}/home``. Idempotent;
    a no-op (``provisioned=False``) when no token is configured.

    Apply-if-absent: never clobbers an existing ``.git-credentials`` /
    ``.gitconfig`` (e.g. a profile's own imported setup) unless ``force``.
    """
    found = _read_env_token(home_dir / ".env")
    if found is None:
        return ProvisionResult(home_dir, False, reason="no GitHub token configured")
    token, source = found

    subprocess_home = home_dir / "home"
    cred_path = subprocess_home / ".git-credentials"
    cfg_path = subprocess_home / ".gitconfig"

    if not force and (cred_path.exists() or cfg_path.exists()):
        return ProvisionResult(
            home_dir,
            False,
            reason="git config already present (pass force to overwrite)",
            source=source,
        )

    resolved_name = name or home_dir.name
    resolved_email = email or f"{resolved_name}@users.noreply.github.com"

    subprocess_home.mkdir(parents=True, exist_ok=True)
    _write_file(cred_path, build_git_credentials_content(token), mode=0o600)
    _write_file(
        cfg_path,
        build_git_config_content(name=resolved_name, email=resolved_email),
        mode=0o644,
    )
    return ProvisionResult(home_dir, True, source=source)


def _write_file(path: Path, content: str, *, mode: int) -> None:
    """Write ``content`` then force ``mode`` (umask-independent)."""
    path.write_text(content, encoding="utf-8")
    os.chmod(path, mode)


def provision_all(hermes_home: Path) -> list[ProvisionResult]:
    """Provision the default profile (HERMES_HOME root) + each named profile.

    The HERMES_HOME root is the implicit default profile, and named profiles
    live under ``{HERMES_HOME}/profiles/<name>/``.

    The root's basename is uninformative in production (``/opt/data`` → "data"),
    so the git identity for the default profile comes from ``HERMES_AGENT_NAME``
    when set — commits are authored as the agent, not "data". A named profile is
    its own agent and is identified by its profile directory name.
    """
    root_name = os.environ.get("HERMES_AGENT_NAME") or None
    results = [provision_git_credentials(hermes_home, name=root_name)]
    profiles_dir = hermes_home / "profiles"
    if profiles_dir.is_dir():
        for profile in sorted(p for p in profiles_dir.iterdir() if p.is_dir()):
            results.append(provision_git_credentials(profile))
    return results


def main() -> int:
    """Entry point invoked during the runtime boot sequence."""
    hermes_home = Path(os.environ.get("HERMES_HOME", "/opt/data"))
    for r in provision_all(hermes_home):
        status = (
            f"provisioned (source={r.source})"
            if r.provisioned
            else f"skipped ({r.reason})"
        )
        print(f"git-credentials: home={r.home} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
