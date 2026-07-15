"""Regression test: ``hermes dump`` reports a real git SHA inside the container.

Background: ``.dockerignore`` excludes ``.git``, so ``git rev-parse HEAD``
fails inside the published image and ``hermes dump`` used to report
``version: ... [(unknown)]``.  The Dockerfile now writes the build-time
``$HERMES_GIT_SHA`` build-arg to ``/opt/hermes/.hermes_build_sha`` and
``hermes_cli/build_info.py`` reads it as a fallback.

CI (``.github/workflows/docker.yml``) always sets the build-arg
to ``${{ github.sha }}``.  Local ``docker build`` (the ``built_image``
fixture in ``tests/docker/conftest.py``) does NOT — so locally the file
is absent and ``hermes dump`` correctly falls back to ``(unknown)``.

This test handles both cases:

* If ``/opt/hermes/.hermes_build_sha`` exists in the image, assert that
  ``hermes dump`` surfaces its content as the version SHA (not
  ``(unknown)``).
* If the file is absent, assert the legacy behaviour (``(unknown)``)
  still holds — defensive guard against the helper accidentally
  reporting bogus data from somewhere else.
"""
from __future__ import annotations

import re
import subprocess

from tests.docker.conftest import docker_exec_sh, wait_for_container_ready


_VERSION_LINE = re.compile(r"^version:\s+(?P<rest>.+)$", re.MULTILINE)
_SHA_BRACKET = re.compile(r"\[(?P<sha>[^\]]+)\]\s*$")


def _run_dump(container: str) -> str:
    """Return the stdout of ``hermes dump`` inside the running container.

    The container is already booted with ``sleep infinity`` by the
    ``shared_container`` fixture, so we just ``docker exec`` the command
    instead of paying the full ``docker run`` startup cost each time.
    """
    r = docker_exec_sh(container, "hermes dump", timeout=60)
    assert r.returncode == 0, (
        f"hermes dump exited {r.returncode}: "
        f"stderr={r.stderr[-1000:]!r}\nstdout={r.stdout[-1000:]!r}"
    )
    return r.stdout


def _read_baked_sha_from_container(container: str) -> str | None:
    """Return the ``/opt/hermes/.hermes_build_sha`` content, or None if absent."""
    r = docker_exec_sh(
        container, "cat /opt/hermes/.hermes_build_sha 2>/dev/null", timeout=10,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return r.stdout.strip() or None


def test_dump_reports_baked_sha_when_present(shared_container: str) -> None:
    """When the image was built with ``HERMES_GIT_SHA``, dump must surface it.

    Together with the smoke-test action (which exercises ``--help``), this
    closes the regression loop for the missing-sha bug: any future change
    that breaks the baked-file -> dump pipeline will fail CI here.
    """
    baked = _read_baked_sha_from_container(shared_container)
    stdout = _run_dump(shared_container)

    match = _VERSION_LINE.search(stdout)
    assert match, f"no `version:` line in dump output:\n{stdout[:2000]}"
    sha_match = _SHA_BRACKET.search(match.group("rest"))
    assert sha_match, (
        f"`version:` line missing [<sha>] bracket: {match.group('rest')!r}"
    )
    reported = sha_match.group("sha")

    if baked is None:
        # Local-build path: no build-arg was passed.  Verify the legacy
        # fallback ``(unknown)`` is intact — guards against the helper
        # ever inventing a SHA from thin air.
        assert reported == "(unknown)", (
            f"expected '(unknown)' when no SHA baked, got {reported!r}"
        )
        return

    # CI path: build-arg was set, baked file exists.  ``hermes dump``
    # truncates to 8 chars via ``git rev-parse --short=8`` semantics.
    assert reported != "(unknown)", (
        "baked SHA file present in image but dump still reported "
        f"'(unknown)' — the build-info fallback is broken.  "
        f"Baked file content: {baked!r}"
    )
    assert reported == baked[:8], (
        f"dump reported {reported!r} but baked file contained {baked!r} "
        f"(expected first 8 chars: {baked[:8]!r})"
    )
