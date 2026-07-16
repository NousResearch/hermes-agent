"""Regression test: ``hermes update`` refuses to apply inside a container.

Phase 5 Task 5.3 (§2.12): the updater's ``apply`` path must refuse inside
a container. The image tag is the version selector — the orchestrator's
job is ``docker pull`` + recreate, not an in-place update.

This test probes two layers:

1. **``is_container()`` returns True** inside the published image. This is
   the low-level container detection (``/.dockerenv``, cgroup markers,
   Kubernetes env). If it ever returns False, the higher-level refusal
   in ``cmd_update`` would also silently break (it relies on the
   ``docker`` install-method stamp, but the stamp is only honoured as
   ``docker`` when ``is_container()`` confirms it — see
   ``detect_install_method()`` in ``hermes_cli/config.py``).

2. **``hermes update`` prints the ``docker pull`` guidance and exits 1.**
   This is the user-facing refusal: the apply path bails before any
   ``git fetch`` / ``subprocess.run`` happens.

Both are exercised at runtime inside the real container image.
"""
from __future__ import annotations

import pytest

from tests.docker.conftest import (
    docker_exec_sh,
    start_container,
)


def test_is_container_returns_true_in_image(
    built_image: str, container_name: str,
) -> None:
    """``is_container()`` must return True inside the published image.

    This is the foundation for the updater refusal: ``detect_install_method``
    ignores a ``docker`` stamp in ``$HERMES_HOME`` when not actually in a
    container (self-healing for shared homes), so ``is_container()`` MUST
    fire for the refusal to hold.
    """
    start_container(built_image, container_name)

    r = docker_exec_sh(
        container_name,
        # Import is_container and check it returns True.
        # The exec shim drops to the hermes user, whose PYTHONPATH
        # includes /opt/hermes/.venv — so hermes_constants is importable.
        "python3 -c \""
        "from hermes_constants import is_container; "
        "print('CONTAINER=%s' % is_container())"
        "\"",
        timeout=15,
    )
    assert r.returncode == 0, (
        f"is_container() probe failed (exit {r.returncode}): "
        f"stdout={r.stdout!r} stderr={r.stderr!r}"
    )
    assert "CONTAINER=True" in r.stdout, (
        f"is_container() returned False inside the container — "
        f"the updater refusal guard is broken: {r.stdout!r}"
    )


@pytest.mark.live_system_guard_bypass
def test_hermes_update_refuses_in_container(
    built_image: str, container_name: str,
) -> None:
    """``hermes update`` must refuse inside the container and print
    ``docker pull`` guidance instead of attempting a git-pull.

    This is the user-facing refusal: ``cmd_update`` checks
    ``detect_install_method() == "docker"`` and bails with
    ``format_docker_update_message()`` + exit 1 before any git commands.

    Regression guard: if the early-return is removed or bypassed, the
    update flow would try ``git fetch`` inside a container that has no
    ``.git`` directory (excluded by ``.dockerignore``) and fail with a
    misleading "Not a git repository" error instead of the actionable
    ``docker pull`` instructions.
    """
    start_container(built_image, container_name)

    r = docker_exec_sh(
        container_name,
        # hermes update should print docker pull guidance and exit 1.
        # We capture both stdout and the exit code.
        "hermes update 2>&1; echo EXIT_CODE=$?",
        timeout=30,
    )
    combined = r.stdout + r.stderr

    # Must mention docker pull (the actionable guidance)
    assert "docker pull" in combined, (
        f"expected 'docker pull' guidance in hermes update output, "
        f"got: {combined!r}"
    )

    # Must have exited non-zero (refusal, not success)
    assert "EXIT_CODE=1" in r.stdout or "EXIT_CODE=" in combined, (
        f"expected non-zero exit from hermes update, got: {r.stdout!r}"
    )
    # Be strict: exit code must be 1
    assert "EXIT_CODE=1" in r.stdout, (
        f"expected exit code 1 from 'hermes update' (refusal), "
        f"got: {r.stdout!r}"
    )

    # Must NOT have attempted a git fetch / git pull
    assert "git fetch" not in combined.lower(), (
        f"hermes update attempted git fetch inside container — "
        f"the refusal guard did not fire: {combined!r}"
    )
    assert "git pull" not in combined.lower(), (
        f"hermes update attempted git pull inside container — "
        f"the refusal guard did not fire: {combined!r}"
    )
