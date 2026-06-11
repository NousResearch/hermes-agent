"""Tests for ``hermes update`` / ``--check`` inside the Docker container.

Background: ``.dockerignore`` excludes ``.git``, so the existing git-pull
update path can never succeed inside the published image.  Before this
fix, ``hermes update`` would fall through to ``"✗ Not a git repository.
Please reinstall: curl ... install.sh"`` — that script installs a *new*
host-side Hermes, not an update to the running container, so the message
was actively misleading.

These tests pin the new behaviour: when ``detect_install_method`` reports
``"docker"`` (stamped by ``docker/stage2-hook.sh``), both the apply path
(``cmd_update``) and the check path (``_cmd_update_check``) print the
``docker pull`` guidance from ``format_docker_update_message`` and exit
with status 1, without running ``git fetch`` / ``subprocess.run``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import _cmd_update_check, cmd_update


# ---------- cmd_update (apply path) ----------


@patch("hermes_cli.config.is_managed", return_value=False)
@patch("hermes_cli.config.detect_install_method", return_value="docker")
@patch("subprocess.run")
def test_cmd_update_in_docker_prints_guidance_and_exits(
    mock_run, _mock_method, _mock_managed, capsys
):
    """``hermes update`` inside Docker → friendly message + exit 1, no git calls."""
    with pytest.raises(SystemExit) as excinfo:
        cmd_update(SimpleNamespace(check=False))

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    # Spot-check the key guidance — exhaustive wording is locked in by the
    # config-module test below to keep these CLI tests resilient to copy edits.
    assert "doesn't apply inside the Docker container" in out
    assert "docker pull nousresearch/hermes-agent:latest" in out

    # No git invocations — the early-return must beat every git command.
    git_calls = [c for c in mock_run.call_args_list if c.args and c.args[0] and "git" in str(c.args[0][0])]
    assert git_calls == [], f"expected no git calls, got: {git_calls}"


@patch("hermes_cli.config.is_managed", return_value=False)
@patch("hermes_cli.config.detect_install_method", return_value="docker")
@patch("subprocess.run")
def test_cmd_update_check_in_docker_prints_guidance_and_exits(
    mock_run, _mock_method, _mock_managed, capsys
):
    """``hermes update --check`` inside Docker → same message + exit 1, no fetch."""
    with pytest.raises(SystemExit) as excinfo:
        cmd_update(SimpleNamespace(check=True, branch=None))

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "doesn't apply inside the Docker container" in out
    assert "docker pull nousresearch/hermes-agent:latest" in out

    git_calls = [c for c in mock_run.call_args_list if c.args and c.args[0] and "git" in str(c.args[0][0])]
    assert git_calls == [], f"expected no git calls, got: {git_calls}"


@patch("hermes_cli.config.is_managed", return_value=False)
@patch("hermes_cli.config.detect_install_method", return_value="docker")
@patch("subprocess.run")
def test_cmd_update_in_docker_ignores_yes_and_force(
    mock_run, _mock_method, _mock_managed, capsys
):
    """``--yes`` / ``--force`` don't bypass the Docker bail-out.

    The point of the bail-out is "git pull will never work here", so even
    a user trying to barge through with ``--yes --force`` should see the
    docker-pull guidance.
    """
    with pytest.raises(SystemExit):
        cmd_update(SimpleNamespace(check=False, yes=True, force=True))

    assert "docker pull" in capsys.readouterr().out
    git_calls = [c for c in mock_run.call_args_list if c.args and c.args[0] and "git" in str(c.args[0][0])]
    assert git_calls == []


# ---------- _cmd_update_check (check path, direct entry) ----------


@patch("hermes_cli.config.detect_install_method", return_value="docker")
@patch("subprocess.run")
def test_cmd_update_check_direct_in_docker(mock_run, _mock_method, capsys):
    """Calling ``_cmd_update_check`` directly (no apply path) also bails."""
    with pytest.raises(SystemExit) as excinfo:
        _cmd_update_check()

    assert excinfo.value.code == 1
    assert "docker pull" in capsys.readouterr().out
    git_calls = [c for c in mock_run.call_args_list if c.args and c.args[0] and "git" in str(c.args[0][0])]
    assert git_calls == []


# ---------- Non-Docker installs unaffected ----------


@patch("hermes_cli.config.is_managed", return_value=False)
@patch("hermes_cli.config.detect_install_method", return_value="git")
@patch(
    "subprocess.run",
    return_value=SimpleNamespace(returncode=0, stdout="0\n", stderr=""),
)
def test_cmd_update_on_git_install_does_not_print_docker_message(
    _mock_run, _mock_method, _mock_managed, capsys
):
    """Source/git installs MUST NOT hit the Docker branch.

    Regression guard: an over-eager detection refactor could accidentally
    route git users through the docker-pull message.  We swallow
    SystemExit / unrelated errors from the rest of the update flow —
    those don't matter for this assertion; what matters is that the
    docker text is absent.

    ``subprocess.run`` is mocked because the git path will otherwise shell
    out to ``git fetch upstream`` / ``git fetch origin`` — on CI runners
    with no ``upstream`` remote configured this can hang past the 30s
    pytest-timeout depending on git's network behaviour.  The stub
    returns a successful CompletedProcess-shaped object with ``"0\\n"``
    stdout, which both keeps the flow shell-free AND parses cleanly as
    the "0 commits behind" rev-list output the check path later parses
    via ``int(rev_result.stdout.strip())``.
    """
    try:
        cmd_update(SimpleNamespace(check=True, branch=None))
    except (SystemExit, Exception):
        # Update flow may exit for unrelated reasons in a stubbed env —
        # that's fine; we only care about the banner not appearing.
        pass

    assert "doesn't apply inside the Docker container" not in capsys.readouterr().out


@patch("hermes_cli.config.detect_install_method", return_value="pip")
@patch("hermes_cli.banner.check_via_pypi", return_value=0)
def test_cmd_update_check_on_pip_install_still_uses_pypi(
    _mock_pypi, _mock_method, capsys
):
    """PyPI installs route to PyPI check, not the Docker bail-out."""
    _cmd_update_check()

    out = capsys.readouterr().out
    assert "Already up to date" in out
    assert "doesn't apply inside the Docker container" not in out


# ---------- format_docker_update_message — content lock ----------


def test_format_docker_update_message_contents():
    """Lock in the high-value content of the Docker update message.

    These are the bits a user actually needs to act on; if any of them
    disappear in a copy edit, the message has lost its value.  Specific
    wording around them is free to evolve (we don't assert full text).
    """
    from hermes_cli.config import format_docker_update_message

    msg = format_docker_update_message()

    # Primary command — the entire reason this message exists.
    assert "docker pull nousresearch/hermes-agent:latest" in msg

    # The four key concepts the message must cover:
    assert "restart" in msg.lower(), "must explain that a restart is required"
    assert "--version" in msg, "must show how to verify the new version"
    assert ":latest" in msg, "must mention tag pinning caveat"
    assert "HERMES_HOME" in msg or "/opt/data" in msg, (
        "must address config persistence across upgrades"
    )

    # Acknowledges that forks exist (build-your-own-image escape hatch).
    assert "fork" in msg.lower() or "Dockerfile" in msg
