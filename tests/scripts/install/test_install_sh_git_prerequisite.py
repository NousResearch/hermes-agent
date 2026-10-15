"""install.sh prerequisites: a git on PATH that cannot run is not git.

macOS ships /usr/bin/git as a stub until Apple's Command Line Tools are
installed, so `command -v git` passes and the clone fails much later with
"git clone failed; no checkout published". On macOS the installer gets git
first: Homebrew when present, else Apple's install dialog, waiting for it
unless a plain headless run cannot.
"""

import os
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "install.sh"
_STUB_ERR = (
    "xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), "
    "missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun"
)
_LICENSE_ERR = (
    "You have not agreed to the Xcode license agreements. "
    "Please run 'sudo xcodebuild -license' from within a Terminal window to review and agree to the Xcode "
    "and Apple SDKs license."
)
_WORKING_GIT = "#!/bin/sh\necho 'git version 2.50.1'\n"


def _exe(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run_prerequisites(tmp_path: Path, *, git: str = "stub", interactive: bool = False, json: bool = False,
                       brew: bool = False, clt_finishes: bool = True, dev: str | None = None,
                       install_rc: int = 0):
    """git: "stub" (fails until the CLT install finishes), "license" (Xcode
    license not accepted) or "working". dev: the active developer directory
    `xcode-select -p` reports: None (unset), "damaged" (a folder a macOS upgrade
    left behind without its tools) or "complete" (holds usr/bin/git).
    install_rc: exit status of `xcode-select --install`."""
    bin_dir = tmp_path / "bin"
    brew_bin = tmp_path / "brew-bin"
    brew_bin.mkdir()
    done = tmp_path / "clt-done"
    calls = tmp_path / "calls.log"
    if git == "working":
        _exe(bin_dir / "git", _WORKING_GIT)
    elif git == "license":
        _exe(bin_dir / "git", f'#!/bin/sh\necho "{_LICENSE_ERR}" >&2\nexit 69\n')
    else:
        _exe(bin_dir / "git", f'#!/bin/sh\nif [ -f "{done}" ]; then echo "git version 2.39.5"; exit 0; fi\n'
                              f'echo "{_STUB_ERR}" >&2\nexit 1\n')
    _exe(bin_dir / "curl", "#!/bin/sh\nexit 0\n")
    dev_dir = tmp_path / "CommandLineTools"
    if dev is not None:
        dev_dir.mkdir()
    if dev == "complete":
        _exe(dev_dir / "usr" / "bin" / "git", _WORKING_GIT)
    _exe(bin_dir / "xcode-select",
         f'#!/bin/sh\nif [ "$1" = -p ]; then\n'
         f'  if [ -d "{dev_dir}" ]; then echo "{dev_dir}"; exit 0; fi\n'
         f'  echo "xcode-select: error: unable to get active developer directory" >&2; exit 2\nfi\n'
         f'echo "xcode-select $*" >> "{calls}"\nexit {install_rc}\n')
    # Each poll interval is where the user may finish Apple's dialog.
    finish = f'touch "{done}"\n' if clt_finishes else ""
    _exe(bin_dir / "sleep", f'#!/bin/sh\necho "sleep $*" >> "{calls}"\n{finish}exit 0\n')
    if brew:
        # Homebrew's bin precedes /usr/bin on PATH, as on a real Mac.
        _exe(bin_dir / "brew", f'#!/bin/sh\necho "brew $* auto_update_off=$HOMEBREW_NO_AUTO_UPDATE" >> "{calls}"\n'
                               f"printf '#!/bin/sh\\necho git version 2.51.0\\n' > \"{brew_bin}/git\"\n"
                               f'chmod +x "{brew_bin}/git"\n')
    probe = (
        'source "$1" --manifest\n'
        f"NON_INTERACTIVE={'false' if interactive else 'true'}\n"
        f"JSON={'true' if json else 'false'}\n"
        f"has_terminal() {{ {'return 0' if interactive else 'return 1'}; }}\n"
        "uv_bootstrap_target() { echo darwin-arm64; }\n"
        "stage_prerequisites\n"
    )
    # System dirs only: a host's real Homebrew must never run from a test.
    path = os.pathsep.join([str(brew_bin), str(bin_dir), "/usr/bin", "/bin"])
    env = dict(os.environ, PATH=path, HOME=str(tmp_path))
    result = subprocess.run(
        ["/bin/bash", "-c", probe, "probe", str(_SCRIPT)],
        env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120, check=False,
    )
    return result, (calls.read_text().splitlines() if calls.is_file() else [])


@pytest.mark.platforms("macos")
def test_macos_stub_waits_for_clt_dialog_then_continues(tmp_path):
    result, calls = _run_prerequisites(tmp_path, interactive=True)

    assert result.returncode == 0, result.stderr
    assert "click Install" in result.stdout
    assert "prerequisites ok" in result.stdout
    assert calls == ["xcode-select --install", "sleep 5"]


@pytest.mark.platforms("macos")
def test_macos_stub_hermes_setup_json_run_waits_too(tmp_path):
    """Hermes-Setup runs stages with --non-interactive --json and no terminal."""
    result, calls = _run_prerequisites(tmp_path, interactive=False, json=True)

    assert result.returncode == 0, result.stderr
    assert "prerequisites ok" in result.stdout
    assert calls == ["xcode-select --install", "sleep 5"]


@pytest.mark.platforms("macos")
def test_macos_stub_prefers_homebrew_without_dialog(tmp_path):
    result, calls = _run_prerequisites(tmp_path, interactive=True, brew=True)

    assert result.returncode == 0, result.stderr
    assert "prerequisites ok" in result.stdout
    assert calls == ["brew install git auto_update_off=1"]


@pytest.mark.platforms("macos")
def test_macos_headless_non_interactive_fails_fast_with_remedy(tmp_path):
    result, calls = _run_prerequisites(tmp_path, interactive=False)

    assert result.returncode != 0
    assert "xcode-select --install" in result.stderr
    assert "prerequisites ok" not in result.stdout
    assert calls == ["xcode-select --install"]


@pytest.mark.platforms("macos")
def test_macos_stub_wait_gives_up_after_fifteen_minutes(tmp_path):
    result, calls = _run_prerequisites(tmp_path, interactive=True, clt_finishes=False)

    assert result.returncode != 0
    assert "did not finish within 15 minutes" in result.stderr
    assert calls.count("sleep 5") == 180
    assert "Still waiting for the Command Line Tools install (15m)" in result.stdout


@pytest.mark.platforms("macos")
def test_macos_tools_present_but_git_fails_reports_git_error_without_waiting(tmp_path):
    """An unaccepted Xcode license: no dialog opens, so neither prompt nor wait."""
    result, calls = _run_prerequisites(tmp_path, git="license", dev="complete", interactive=True)

    assert result.returncode != 0
    assert "git is installed but cannot run" in result.stderr
    assert "sudo xcodebuild -license" in result.stderr
    assert "Command Line Tools" not in result.stderr
    assert calls == []


@pytest.mark.platforms("macos")
def test_macos_upgrade_leftover_tools_folder_still_opens_dialog(tmp_path):
    """After a macOS upgrade `-p` names a folder that exists but holds no git."""
    result, calls = _run_prerequisites(tmp_path, dev="damaged", interactive=True)

    assert result.returncode == 0, result.stderr
    assert "prerequisites ok" in result.stdout
    assert calls == ["xcode-select --install", "sleep 5"]


@pytest.mark.platforms("macos")
def test_macos_damaged_tools_install_refused_fails_fast_with_reinstall_remedy(tmp_path):
    """`--install` refuses over a leftover folder: no dialog opens, so no wait."""
    result, calls = _run_prerequisites(tmp_path, dev="damaged", install_rc=1, interactive=True)

    assert result.returncode != 0
    assert "sudo rm -rf /Library/Developer/CommandLineTools" in result.stderr
    assert "prerequisites ok" not in result.stdout
    assert calls == ["xcode-select --install"]


@pytest.mark.platforms("linux")
def test_broken_git_on_linux_names_git_error_without_xcode(tmp_path):
    result, calls = _run_prerequisites(tmp_path)

    assert result.returncode != 0
    assert "git is installed but cannot run" in result.stderr
    assert "invalid active developer path" in result.stderr
    assert calls == []


@pytest.mark.platforms("posix")
def test_working_git_passes_prerequisites(tmp_path):
    result, calls = _run_prerequisites(tmp_path, git="working")

    assert result.returncode == 0, result.stderr
    assert "prerequisites ok" in result.stdout
    assert calls == []
