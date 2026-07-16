"""``hermes adopt`` subcommand — hop 3 of the adoption funnel.

Fetches the platform ``hermes-updater`` binary, execs it with
``adopt --from-checkout <PROJECT_ROOT>``, and never returns (``os.execv``
replaces the process image).  The updater handles the actual slot
creation, symlink re-point, and data-dir state seeding.

See ``docs/plans/updater-rework/03-phase2-compat-and-adoption.md`` task 2.5.
"""

from __future__ import annotations

import hashlib
import os
import platform
import stat
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RELEASE_BASE = (
    "https://github.com/NousResearch/hermes-agent/releases/latest/download"
)

# Platform suffixes for the updater binary names.  ``hermes-updater-<suffix>``.
def _platform_suffix() -> str:
    """Return the release-asset suffix for the current platform."""
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        return f"darwin-{arch}"
    if sys.platform == "win32":
        arch = "x86_64" if machine in ("amd64", "x86_64") else "aarch64"
        return f"windows-{arch}.exe"
    # Linux and other POSIX — include musl/glibc distinction later if needed.
    arch = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
    return f"linux-{arch}"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_updater(
    dest: Path,
    *,
    source_base: Optional[str] = None,
) -> Path:
    """Download the platform ``hermes-updater`` binary to *dest*.

    If *source_base* is given (``https://`` or ``file://``), it replaces
    the default GitHub release base URL.  The binary name is
    ``hermes-updater-<platform-suffix>``.

    On success, *dest* is made executable and returned.
    """
    suffix = _platform_suffix()
    base = source_base or DEFAULT_RELEASE_BASE
    url = f"{base.rstrip('/')}/hermes-updater-{suffix}"

    dest.parent.mkdir(parents=True, exist_ok=True)

    # urllib.request supports both http(s):// and file:// schemes.
    with urllib.request.urlopen(url) as resp:  # noqa: S310 — URL is constructed
        data = resp.read()

    dest.write_bytes(data)
    dest.chmod(dest.stat().st_mode | stat.S_IRWXU)
    return dest


def _verify_sha256(path: Path, expected: str) -> bool:
    """Return True if *path*'s sha256 matches *expected*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest() == expected.lower()


# ---------------------------------------------------------------------------
# Install-method guards
# ---------------------------------------------------------------------------

# Methods that should use their package-manager / image upgrade path instead
# of adoption.
_BLOCKED_METHODS = frozenset({"docker", "nixos", "homebrew", "pip"})

# Human-readable recommended commands for each blocked method.
_RECOMMENDED_COMMANDS: dict[str, str] = {
    "docker": "docker pull nousresearch/hermes-agent:latest",
    "nixos": "Update your Nix flake input and rebuild (e.g. nix flake update, nixos-rebuild)",
    "homebrew": "brew upgrade hermes-agent",
    "pip": "pip install --upgrade hermes-agent",
}


def _refuse_blocked_method(method: str) -> None:
    """Print the recommended-command message and exit(1) for blocked methods."""
    cmd = _RECOMMENDED_COMMANDS.get(method)
    if cmd:
        print(
            f"Cannot adopt: this Hermes installation uses the '{method}' "
            f"install method.\n"
            f"Use your package manager / image update path instead:\n"
            f"  {cmd}\n",
            file=sys.stderr,
        )
    else:
        print(
            f"Cannot adopt: this Hermes installation uses the '{method}' "
            f"install method, which is not compatible with adoption.\n",
            file=sys.stderr,
        )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dirty-tree guard
# ---------------------------------------------------------------------------

def _print_dirty_warning(reasons: list[str]) -> None:
    """Print the eject-vs-adopt choice for dirty/fork cohorts."""
    print(
        "\n⚠  Your checkout is not a pristine upstream install:\n",
        file=sys.stderr,
    )
    for r in reasons:
        print(f"    • {r}", file=sys.stderr)
    print(
        "\n"
        "Adoption will switch to managed releases (faster, atomic,\n"
        "rollbackable updates — no local building).  Your current checkout\n"
        "is kept untouched as a fallback.\n"
        "\n"
        "If you want to keep developing from this checkout, use `git` to\n"
        "pull updates instead:\n"
        "  git pull\n"
        "\n"
        "To proceed with adoption anyway, re-run with --yes-dirty.\n",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

def cmd_adopt(args) -> None:
    """``hermes adopt`` — download the updater, exec it, never return.

    (a) Refuses for docker/nix/brew/pip.
    (b) Requires ``--yes-dirty`` for dirty/fork trees.
    (c) Downloads the platform ``hermes-updater`` to ``$HERMES_HOME/bin/``.
    (d) ``os.execv``s it — Python never returns.
    """
    from hermes_cli.config import detect_install_method
    from hermes_constants import get_hermes_home

    # Resolve project root from main (avoids a circular import at module load).
    from hermes_cli.main import PROJECT_ROOT

    # --- (a) Refuse blocked install methods ---
    method = detect_install_method(PROJECT_ROOT)
    if method in _BLOCKED_METHODS:
        _refuse_blocked_method(method)

    # --- (b) Dirty/fork cohort guard ---
    yes_dirty = getattr(args, "yes_dirty", False)
    try:
        from hermes_cli.adoption import detect_legacy_install
        legacy = detect_legacy_install(PROJECT_ROOT, get_hermes_home())
    except Exception:
        # Detector is crash-proof: if it can't run, skip the dirty guard
        # rather than blocking adoption.  (See plan task 2.4 / 2.5.)
        legacy = None

    if legacy is not None and not legacy.pristine:
        if not yes_dirty:
            _print_dirty_warning(legacy.reasons)
            sys.exit(1)

    # --- (c) Download the updater binary ---
    hermes_home = get_hermes_home()
    bin_dir = hermes_home / "bin"
    updater_name = "hermes-updater"
    if sys.platform == "win32":
        updater_name = "hermes-updater.exe"
    updater_path = bin_dir / updater_name

    source_url = getattr(args, "source", None)
    _download_updater(updater_path, source_base=source_url)

    if not updater_path.exists():
        print(f"Error: failed to download hermes-updater to {updater_path}",
              file=sys.stderr)
        sys.exit(1)

    # --- (d) execv the updater — never returns ---
    # Build argv: ["hermes-updater", "adopt", "--from-checkout", PROJECT_ROOT,
    #              "--source", <url> (if provided)]
    updater_argv = [
        "hermes-updater",
        "adopt",
        "--from-checkout",
        str(PROJECT_ROOT),
    ]
    if source_url:
        updater_argv += ["--source", source_url]

    # os.execv replaces the process image — Python never returns from here.
    os.execv(str(updater_path), updater_argv)

    # Unreachable under normal conditions.  If execv fails (e.g. permission
    # denied), it raises OSError.  If somehow we get here, it's a bug.
    raise RuntimeError("os.execv returned — this should never happen")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_adopt_parser(subparsers, *, cmd_adopt: Callable) -> None:
    """Attach the ``adopt`` subcommand to ``subparsers``."""
    adopt_parser = subparsers.add_parser(
        "adopt",
        help="Switch this install to managed releases (slot-based)",
        description=(
            "Download the hermes-updater and hand off to it.  The updater "
            "creates a managed slot, re-points the PATH symlink, and leaves "
            "your current checkout untouched as a fallback.  This command "
            "does not return — the updater process replaces this one."
        ),
    )
    adopt_parser.add_argument(
        "--source",
        default=None,
        metavar="URL",
        help=(
            "Base URL for downloading the hermes-updater binary "
            "(https:// or file://).  Default: latest GitHub release."
        ),
    )
    adopt_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Skip interactive prompts.",
    )
    adopt_parser.add_argument(
        "--yes-dirty",
        action="store_true",
        default=False,
        help=(
            "Force adoption even when the checkout is dirty or a fork. "
            "Without this flag, adoption of a non-pristine tree is refused."
        ),
    )
    adopt_parser.set_defaults(func=cmd_adopt)
