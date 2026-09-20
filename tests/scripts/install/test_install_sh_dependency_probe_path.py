"""Regression tests for the install.sh dependency-probe PATH.

A pre-staged static ripgrep/ffmpeg in the installer's own command link dir
(~/.local/bin for user installs, $PREFIX/bin on Termux, /usr/local/bin for
root FHS) must be detected by install_system_packages: main() probes there
before setup_path has put the dir on PATH, so a bare `command -v` misses the
binary and the user is offered a heavy distro package they deliberately
avoided. See #116809.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _install_system_packages_body() -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    head, _, rest = text.partition("install_system_packages() {\n")
    assert rest, "Could not find install_system_packages() in scripts/install.sh"
    body, _, _ = rest.partition("\n}\n")
    assert body, "Could not find install_system_packages() closing brace"
    return body


def test_probe_path_appends_command_link_dir() -> None:
    body = _install_system_packages_body()

    assert 'command_link_dir="$(get_command_link_dir 2>/dev/null)"' in body
    # Only append when the dir exists and is not already on PATH.
    assert '[ -d "$command_link_dir" ]' in body
    assert 'probe_path="$PATH:$command_link_dir"' in body


def test_dependency_probes_use_probe_path() -> None:
    body = _install_system_packages_body()

    # Both probes must run against the extended PATH, not the bare one.
    assert 'PATH="$probe_path" command -v rg' in body
    assert 'PATH="$probe_path" command -v ffmpeg' in body
    # Version probes follow the same resolution as the detection probes.
    assert 'PATH="$probe_path" rg --version' in body
    assert 'PATH="$probe_path" ffmpeg -version' in body
