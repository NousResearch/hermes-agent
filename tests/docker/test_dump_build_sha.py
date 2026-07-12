"""Docker artifacts expose the same exact source revision everywhere.

``.dockerignore`` excludes ``.git``, so the image must rely on immutable
package-relative build metadata. CI passes ``$HERMES_GIT_SHA`` and the
Dockerfile writes it to ``hermes_cli/_build_metadata.json``; local builds
without the argument carry an explicit null revision.

CI (``.github/workflows/docker.yml``) always sets the build-arg
to ``${{ github.sha }}``. The local ``built_image`` fixture passes its clean
checkout revision, or omits the argument when the checkout is dirty.

The test reads the artifact file, resolves the public build-info API with no
Git directory, and checks the legacy short ``hermes dump`` display derives
from that same full revision.
"""
from __future__ import annotations

import json
import os
import re
import subprocess


_VERSION_LINE = re.compile(r"^version:\s+(?P<rest>.+)$", re.MULTILINE)
_SHA_BRACKET = re.compile(r"\[(?P<sha>[^\]]+)\]\s*$")


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


def _read_baked_metadata_from_image(image: str) -> dict:
    r = subprocess.run(
        [
            "docker", "run", "--rm", "--entrypoint", "cat", image,
            "/opt/hermes/hermes_cli/_build_metadata.json",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _resolve_source_revision_in_image(image: str) -> str:
    r = subprocess.run(
        [
            "docker", "run", "--rm",
            "--entrypoint", "/opt/hermes/.venv/bin/python", image,
            "-c",
            "from hermes_cli.build_info import get_source_revision; "
            "print(get_source_revision())",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_docker_build_identity_matches_dump(
    built_image: str,
    built_image_source_revision: str | None,
) -> None:
    """When the image was built with ``HERMES_GIT_SHA``, dump must surface it.

    Together with the smoke-test action (which exercises ``--help``), this
    closes the regression loop for the missing-sha bug: any future change
    that breaks the baked-file -> dump pipeline will fail CI here.
    """
    metadata = _read_baked_metadata_from_image(built_image)
    assert set(metadata) == {"source_revision"}
    baked = metadata["source_revision"]
    assert baked is None or re.fullmatch(r"[0-9a-f]{40}", baked)
    if not os.environ.get("HERMES_TEST_IMAGE"):
        assert baked == built_image_source_revision

    resolved = _resolve_source_revision_in_image(built_image)
    stdout = _run_dump(built_image)

    match = _VERSION_LINE.search(stdout)
    assert match, f"no `version:` line in dump output:\n{stdout[:2000]}"
    sha_match = _SHA_BRACKET.search(match.group("rest"))
    assert sha_match, (
        f"`version:` line missing [<sha>] bracket: {match.group('rest')!r}"
    )
    reported = sha_match.group("sha")

    if baked is None:
        assert resolved == "None"
        assert reported == "(unknown)", (
            f"expected '(unknown)' when no SHA baked, got {reported!r}"
        )
        return

    assert resolved == baked
    assert reported != "(unknown)", (
        "build metadata present in image but dump still reported "
        f"'(unknown)' — the build-info fallback is broken.  "
        f"Metadata revision: {baked!r}"
    )
    assert reported == baked[:8], (
        f"dump reported {reported!r} but metadata contained {baked!r} "
        f"(expected first 8 chars: {baked[:8]!r})"
    )
