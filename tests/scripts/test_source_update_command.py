"""Source E2E updates must follow the staged git branch when supported."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


HELPER = Path(__file__).resolve().parents[1] / "install/e2e-assets/source-update-command.sh"
SOURCE_BUILD_ENV = HELPER.with_name("source-build-env.sh")
REPO_ROOT = HELPER.parents[3]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    ("help_text", "expected"),
    [
        ("update [--yes] [--branch NAME]", "update --yes --branch main"),
        ("update [--yes]", "update --yes"),
        ("update [--switch-branch]", "update"),
        ("update", "update"),
    ],
)
def test_source_update_follows_staged_main_if_installed_cli_accepts_branch(tmp_path, help_text, expected):
    cli = tmp_path / "installed hermes"
    cli.write_text('''#!/usr/bin/env bash
if [[ "$*" == *"--branch main"* ]]; then
  printf 'selected staged main: %s\\n' "$*"
elif [[ "$EXPECT_BRANCH" == 1 ]]; then
  printf 'Channel object not found: releases/channels/main.json\\n' >&2
  exit 1
else
  printf 'legacy updater: %s\\n' "$*"
fi
''', encoding="utf-8")
    cli.chmod(0o755)
    result = subprocess.run(
        ["bash", "-euc", 'source "$HELPER"; build_source_update_command "$CLI" "$HELP_TEXT"; "${update_cmd[@]}"'],
        env=dict(os.environ, HELPER=str(HELPER), CLI=str(cli), HELP_TEXT=help_text,
                 EXPECT_BRANCH="1" if "--branch NAME" in help_text else "0"),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith(expected)
    assert "Channel object not found" not in result.stderr


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # --help exits in argparse, before the update handler.
def test_installed_cli_help_drives_explicit_staged_branch(tmp_path):
    # Use the real update parser's help, not a synthetic one-line help fixture.
    # The fake executable only replaces the destructive update action.
    env = dict(os.environ, HERMES_DISABLE_LAZY_INSTALLS="1", PYTHONDONTWRITEBYTECODE="1")
    help_result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "update", "--help"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert help_result.returncode == 0, help_result.stderr
    cli = tmp_path / "installed hermes"
    cli.write_text('''#!/usr/bin/env bash
[[ "$1" == update ]] || exit 2
shift
[[ " $* " == *" --yes "* && " $* " == *" --branch main "* ]] || {
  printf 'Channel object not found: releases/channels/main.json\\n' >&2
  exit 1
}
printf 'selected staged main: %s\\n' "$*"
''', encoding="utf-8")
    cli.chmod(0o755)
    result = subprocess.run(
        ["bash", "-euc", 'source "$HELPER"; source "$SOURCE_BUILD_ENV"; '
         'build_source_update_command "$CLI" "$HELP_TEXT"; source_build_env "${update_cmd[@]}"'],
        env=dict(env, HELPER=str(HELPER), SOURCE_BUILD_ENV=str(SOURCE_BUILD_ENV),
                 CLI=str(cli), HELP_TEXT=help_result.stdout),
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "selected staged main: --yes --branch main" in result.stdout
