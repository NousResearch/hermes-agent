"""Regression test: ``hermes dump`` reports a real git SHA inside the container.

``.dockerignore`` excludes ``.git``, so ``git rev-parse HEAD`` fails inside
the published image. CI writes ``install-stamp.json`` before ``docker build``
(scripts/write_install_stamp.py) and it is COPY'd to the canonical
``/opt/hermes/install-stamp.json``. ``hermes dump`` reads the
commit from that stamp through ``hermes_cli.version_info``.

A local ``docker build`` (the ``built_image`` fixture in
``tests/docker/conftest.py``) has no CI stamp — only the Dockerfile's
distribution-only fallback, whose all-zero commit version_info skips. In
that case ``hermes dump`` falls back to ``(unknown)``.

This test asserts both cases:

* When the stamp exists in the image, ``hermes dump`` must show the first 8
  characters of its commit, not ``(unknown)``.
* When the stamp is absent, ``hermes dump`` must show ``(unknown)`` — a guard
  against the helper inventing a SHA from another source.
"""
from __future__ import annotations

import json
import re
import subprocess


_VERSION_LINE = re.compile(r"^version:\s+(?P<rest>.+)$", re.MULTILINE)
# The stamp-first version line is "X.Y.Z [<sha>] (YYYY-MM-DD)" — the sha
# bracket may be followed by the build-date suffix.
_SHA_BRACKET = re.compile(r"\[(?P<sha>[^\]]+)\]")


def _run_dump(image: str) -> str:
    """Return the stdout of ``docker run <image> dump``.

    Relies on Docker's anonymous VOLUME for ``/opt/data`` (declared by the
    Dockerfile) so the container's hermes user (UID 10000) can bootstrap
    its config.  Anonymous volumes are auto-cleaned by ``--rm``, so unlike
    a host bind-mount we don't have to chown anything to UID 10000 (which
    would break cleanup on non-root hosts).
    """
    r = subprocess.run(
        ["docker", "run", "--rm", image, "dump"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, (
        f"hermes dump exited {r.returncode}: "
        f"stderr={r.stderr[-1000:]!r}\nstdout={r.stdout[-1000:]!r}"
    )
    return r.stdout


def _read_stamp_commit_from_image(image: str) -> str | None:
    """Return the stamp commit from the image, or None when absent/unusable."""
    r = subprocess.run(
        [
            "docker", "run", "--rm", "--entrypoint", "cat", image,
            "/opt/hermes/install-stamp.json",
        ],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        return None
    try:
        commit = json.loads(r.stdout).get("commit") or ""
    except ValueError:
        return None
    # An all-zero commit is the writer's fallback placeholder, and
    # version_info skips it the same way.
    if not commit or set(commit) == {"0"}:
        return None
    return commit


def test_dump_reports_stamp_commit_when_present(built_image: str) -> None:
    """When the image carries an install stamp, dump must surface its commit.

    Together with the smoke-test action (which exercises ``--help``), this
    closes the regression loop for the missing-sha bug: any future change
    that breaks the stamp -> dump pipeline will fail CI here.
    """
    stamped = _read_stamp_commit_from_image(built_image)
    stdout = _run_dump(built_image)

    match = _VERSION_LINE.search(stdout)
    assert match, f"no `version:` line in dump output:\n{stdout[:2000]}"
    sha_match = _SHA_BRACKET.search(match.group("rest"))
    assert sha_match, (
        f"`version:` line missing [<sha>] bracket: {match.group('rest')!r}"
    )
    reported = sha_match.group("sha")

    if stamped is None:
        # Local-build path: no stamp in the image. The fallback must stay
        # '(unknown)' — a guard against the helper inventing a SHA.
        assert reported == "(unknown)", (
            f"expected '(unknown)' when no stamp is baked, got {reported!r}"
        )
        return

    # CI path: the stamp exists. ``hermes dump`` shows the first 8 chars.
    assert reported != "(unknown)", (
        "install stamp present in image but dump still reported "
        f"'(unknown)' — the stamp fallback is broken. Stamp commit: {stamped!r}"
    )
    assert reported == stamped[:8], (
        f"dump reported {reported!r} but the stamp commit is {stamped!r} "
        f"(expected first 8 chars: {stamped[:8]!r})"
    )
