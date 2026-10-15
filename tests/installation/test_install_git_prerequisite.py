"""install.sh prerequisites: a git on PATH that cannot run is not git.

macOS ships /usr/bin/git as a stub until Apple's Command Line Tools are
installed, so `command -v git` passes and the clone fails much later with
"git clone failed; no checkout published". On macOS the installer gets git
first: Homebrew when present, else Apple's install dialog, waiting for it
when a terminal is attached.
"""

import os
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install.sh"
_STUB_ERR = (
    "xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), "
    "missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun"
)
_WORKING_GIT = "#!/bin/sh\necho 'git version 2.50.1'\n"


def _exe(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run_prerequisites(tmp_path: Path, *, system: str, git: str = "stub", interactive: bool = False,
                       brew: bool = False, clt_finishes: bool = True):
    """git: "stub" (fails until the CLT install finishes) or "working"."""
    bin_dir = tmp_path / "bin"
    brew_bin = tmp_path / "brew-bin"
    brew_bin.mkdir()
    done = tmp_path / "clt-done"
    calls = tmp_path / "calls.log"
    if git == "working":
        _exe(bin_dir / "git", _WORKING_GIT)
    else:
        _exe(bin_dir / "git", f'#!/bin/sh\nif [ -f "{done}" ]; then echo "git version 2.39.5"; exit 0; fi\n'
                              f'echo "{_STUB_ERR}" >&2\nexit 1\n')
    _exe(bin_dir / "curl", "#!/bin/sh\nexit 0\n")
    _exe(bin_dir / "uname", f"#!/bin/sh\necho {system}\n")
    _exe(bin_dir / "xcode-select", f'#!/bin/sh\necho "xcode-select $*" >> "{calls}"\nexit 1\n')
    # Each poll interval is where the user may finish Apple's dialog.
    finish = f'touch "{done}"\n' if clt_finishes else ""
    _exe(bin_dir / "sleep", f'#!/bin/sh\necho "sleep $*" >> "{calls}"\n{finish}exit 0\n')
    if brew:
        # Homebrew's bin precedes /usr/bin on PATH, as on a real Mac.
        _exe(bin_dir / "brew", f'#!/bin/sh\necho "brew $*" >> "{calls}"\n'
                               f"printf '#!/bin/sh\\necho git version 2.51.0\\n' > \"{brew_bin}/git\"\n"
                               f'chmod +x "{brew_bin}/git"\n')
    probe = (
        'source "$1" --manifest\n'
        f"NON_INTERACTIVE={'false' if interactive else 'true'}\n"
        f"has_terminal() {{ {'return 0' if interactive else 'return 1'}; }}\n"
        "uv_bootstrap_target() { echo darwin-arm64; }\n"
        "stage_prerequisites\n"
    )
    path = os.pathsep.join([str(brew_bin), str(bin_dir), os.environ["PATH"]])
    env = dict(os.environ, PATH=path, HOME=str(tmp_path))
    result = subprocess.run(
        ["bash", "-c", probe, "probe", str(_SCRIPT)],
        env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120, check=False,
    )
    return result, (calls.read_text().splitlines() if calls.is_file() else [])


@pytest.mark.platforms("posix")
def test_macos_stub_waits_for_clt_dialog_then_continues(tmp_path):
    result, calls = _run_prerequisites(tmp_path, system="Darwin", interactive=True)

    assert result.returncode == 0, result.stderr
    assert "click Install" in result.stdout
    assert "prerequisites ok" in result.stdout
    assert calls == ["xcode-select --install", "sleep 5"]


@pytest.mark.platforms("posix")
def test_macos_stub_prefers_homebrew_without_dialog(tmp_path):
    result, calls = _run_prerequisites(tmp_path, system="Darwin", interactive=True, brew=True)

    assert result.returncode == 0, result.stderr
    assert "prerequisites ok" in result.stdout
    assert calls == ["brew install git"]


@pytest.mark.platforms("posix")
def test_macos_stub_non_interactive_fails_fast_with_remedy(tmp_path):
    result, calls = _run_prerequisites(tmp_path, system="Darwin", interactive=False)

    assert result.returncode != 0
    assert "xcode-select --install" in result.stderr
    assert "prerequisites ok" not in result.stdout
    assert calls == ["xcode-select --install"]


@pytest.mark.platforms("posix")
def test_macos_stub_wait_gives_up_after_fifteen_minutes(tmp_path):
    result, calls = _run_prerequisites(tmp_path, system="Darwin", interactive=True, clt_finishes=False)

    assert result.returncode != 0
    assert "did not finish within 15 minutes" in result.stderr
    assert calls.count("sleep 5") == 180
    assert "Still waiting for the Command Line Tools install (15m)" in result.stdout


@pytest.mark.platforms("posix")
def test_broken_git_elsewhere_names_git_error_without_xcode(tmp_path):
    result, calls = _run_prerequisites(tmp_path, system="Linux")

    assert result.returncode != 0
    assert "git is installed but cannot run" in result.stderr
    assert "invalid active developer path" in result.stderr
    assert calls == []


@pytest.mark.platforms("posix")
def test_working_git_passes_prerequisites(tmp_path):
    result, calls = _run_prerequisites(tmp_path, system="Darwin", git="working")

    assert result.returncode == 0, result.stderr
    assert "prerequisites ok" in result.stdout
    assert calls == []
