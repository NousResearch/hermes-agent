"""install.sh must stamp the desktop bootstrap-complete marker.

The marker at ``$INSTALL_DIR/.hermes-bootstrap-complete`` is what the desktop
app (apps/desktop/electron/main.ts) and the macOS launcher fast path
(apps/bootstrap-installer) use to decide "a real install finished here."
install.sh never wrote it, so a CLI-installed Mac/Linux box re-ran first-run
bootstrap on every desktop launch (#60721).

These exercise the real shell function against a temp checkout rather than
asserting on the text of install.sh.
"""

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def run_write_marker(install_dir, *, commit="", branch="main"):
    """Source install.sh and invoke write_bootstrap_marker in isolation.

    install.sh guards its own entrypoint behind MANIFEST_MODE/STAGE_NAME/main,
    so sourcing it with --help-less argv defines the functions without running
    an install.
    """
    script = f"""
set -e
INSTALL_DIR={install_dir!s}
INSTALL_COMMIT={commit!r}
BRANCH={branch!r}
# Pull in the function definitions without triggering an install.
eval "$(sed -n '/^write_bootstrap_marker()/,/^}}/p' {INSTALL_SH!s})"
log_warn() {{ echo "WARN: $*" >&2; }}
write_bootstrap_marker
"""
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30
    )


def make_checkout(tmp_path):
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=install_dir, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "--allow-empty", "-m", "init"],
        cwd=install_dir,
        check=True,
    )
    return install_dir


def test_marker_matches_the_schema_the_desktop_validates(tmp_path):
    """Desktop's isBootstrapComplete() needs schemaVersion 1 + a >=7 char commit."""
    install_dir = make_checkout(tmp_path)

    result = run_write_marker(install_dir)
    assert result.returncode == 0, result.stderr

    marker = install_dir / ".hermes-bootstrap-complete"
    assert marker.is_file(), "install.sh must stamp the bootstrap marker"

    payload = json.loads(marker.read_text())
    assert payload["schemaVersion"] == 1
    assert len(payload["pinnedCommit"]) >= 7
    assert payload["pinnedBranch"] == "main"
    assert payload["completedAt"].endswith("Z")


def test_marker_publish_leaves_no_temp_sibling(tmp_path):
    """The launcher predicate is existence-only, so the write must be atomic."""
    install_dir = make_checkout(tmp_path)

    run_write_marker(install_dir)

    assert (install_dir / ".hermes-bootstrap-complete").is_file()
    assert not (install_dir / ".hermes-bootstrap-complete.tmp").exists()


def test_explicit_commit_pin_wins_over_head(tmp_path):
    install_dir = make_checkout(tmp_path)
    pinned = "abcdef1234567890abcdef1234567890abcdef12"

    run_write_marker(install_dir, commit=pinned)

    payload = json.loads((install_dir / ".hermes-bootstrap-complete").read_text())
    assert payload["pinnedCommit"] == pinned


def test_no_marker_written_when_head_cannot_be_resolved(tmp_path):
    """A malformed marker is worse than none: absent means a clean re-bootstrap."""
    install_dir = tmp_path / "not-a-checkout"
    install_dir.mkdir()

    result = run_write_marker(install_dir)

    assert result.returncode == 0, "an unresolvable HEAD must not fail the install"
    assert not (install_dir / ".hermes-bootstrap-complete").exists()


def test_missing_install_dir_is_not_fatal(tmp_path):
    result = run_write_marker(tmp_path / "does-not-exist")

    assert result.returncode == 0
