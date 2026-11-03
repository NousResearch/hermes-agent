"""The stored-credential lookup behind private plugin installs never waits on a window.

Regression for #140981: PortableGit, which Hermes ships on Windows, configures
``credential.helper = helper-selector`` system-wide. That helper opens a picker window on every call,
so ``git credential fill`` blocked until someone clicked it. Its ``subprocess.run`` timeout never fired
either: on Windows ``run()`` waits for the captured pipes after killing git, and the helper's
processes still hold them.

The helpers here are shell stubs configured by absolute path. Git runs helpers through ``sh`` on every
OS, and an absolute path keeps a real ``git-credential-helper-selector.exe`` on PATH out of reach.
"""

import time
from pathlib import Path

import pytest

from hermes_cli import git_credentials

ORIGIN = "https://git.example.test"


@pytest.fixture
def gitconfig(tmp_path, monkeypatch) -> Path:
    """The only git config the lookup sees: no system file, this global file, a scratch cwd."""
    path = tmp_path / "gitconfig"
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(path))
    monkeypatch.chdir(tmp_path)
    return path


def _helper(directory: Path, name: str, body: str) -> str:
    """Write an executable shell helper and return the absolute path git config should name."""
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8", newline="\n")
    path.chmod(0o755)
    return path.as_posix()


def _selector(directory: Path) -> tuple[str, Path]:
    """A stand-in for Git for Windows' picker: records that it was started, answers nothing."""
    marker = directory / "selector-started"
    return _helper(directory, "git-credential-helper-selector", f"touch '{marker.as_posix()}'"), marker


def _answering(directory: Path, name: str, password: str) -> str:
    return _helper(directory, name, f"[ \"$1\" = get ] && printf 'username=bob\\npassword={password}\\n'")


@pytest.mark.platforms("any")
def test_helper_selector_is_never_started_and_the_other_helpers_still_answer(tmp_path, gitconfig):
    selector, started = _selector(tmp_path)
    stored = _answering(tmp_path, "stored-helper", "pw-from-store")
    gitconfig.write_text(
        f"[credential]\n\thelper = {selector}\n[credential \"{ORIGIN}\"]\n\thelper = {stored}\n",
        encoding="utf-8")

    assert git_credentials._credential_fill(ORIGIN) == ("bob", "pw-from-store")
    assert not started.exists(), "the picker window must never be opened by a non-interactive lookup"


@pytest.mark.platforms("any")
@pytest.mark.parametrize(("choice", "expected"), [
    ("picked-helper", ("bob", "pw-from-picked")),
    ("<no helper>", None),
], ids=["picked-helper-answers", "no-helper-picked"])
def test_helper_selector_is_swapped_for_the_choice_made_in_it(tmp_path, gitconfig, choice, expected):
    """``credential.helperselector.selected`` is what the picker would pre-tick: that helper is
    asked directly, and the picker's own "no helper" choice means no helper at all."""
    selector, started = _selector(tmp_path)
    picked = _answering(tmp_path, "picked-helper", "pw-from-picked")
    selected = picked if choice == "picked-helper" else choice
    gitconfig.write_text(
        f"[credential]\n\thelper = {selector}\n[credential \"helperselector\"]\n\tselected = {selected}\n",
        encoding="utf-8")

    assert git_credentials._credential_fill(ORIGIN) == expected
    assert not started.exists()


@pytest.mark.platforms("any")
def test_a_helper_that_never_answers_is_stopped_at_the_lookup_timeout(tmp_path, gitconfig):
    """The lookup timeout must end the call even while the helper's processes hold git's pipes."""
    stuck = _helper(tmp_path, "stuck-helper", "sleep 60")
    gitconfig.write_text(f"[credential]\n\thelper = {stuck}\n", encoding="utf-8")

    started = time.monotonic()
    assert git_credentials._credential_fill(ORIGIN) is None
    assert time.monotonic() - started < 45
