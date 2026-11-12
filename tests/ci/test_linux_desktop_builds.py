"""The Linux AppImage builds run on tag builds only and stage to the tag archive, never to a feed."""
import json
import os
import re
import subprocess

import pytest

from scripts.releases.job_groups import DIAGNOSTIC_JOBS
from tests.ci.desktop_release_roles import (
    CANARY_TAG, DOWNLOADABLE_DISPATCHES, DRY_DISPATCH, SHA, admitted, gate, linux_builds,
)
from tests.ci.test_commit_build_staging import shell_step
from tests.ci.test_desktop_release_tag_admission import _workflow
from tests.scripts.test_release_r2 import r2_server


def _stage_step(job):
    (step,) = [step for step in job["steps"] if "scripts.releases.handoff stage" in step.get("run", "")]
    return step


# ALSA's runtime is libasound2 on 22.04; 24.04 renamed it libasound2t64 and
# left libasound2 a virtual package that apt refuses to pick a provider for.
_ALSA = {"ubuntu:22.04": "libasound2", "ubuntu:24.04": "libasound2t64"}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("target", ["linux-x64", "linux-arm64"])
def test_linux_userland_can_load_electron(tmp_path, target):
    """Preparation's native probe starts the downloaded Electron, which won't load without its libraries."""
    jobs = _workflow()["jobs"]
    job = jobs[linux_builds(jobs)[target]]
    (row,) = job["strategy"]["matrix"]["target"]
    (step,) = [step for step in job["steps"] if step.get("name") == "Install the build userland"]
    env = {key: re.sub(r"\$\{\{\s*matrix\.target\.(\w+)\s*\}\}", lambda m: row[m[1]], value)
           for key, value in step.get("env", {}).items()}
    stubs, installed = tmp_path / "bin", tmp_path / "installed"
    stubs.mkdir()
    for name in ("apt-get", "ln", "git"):
        (stubs / name).write_text(
            '#!/bin/sh\nif [ "$1" = install ]; then printf "%s\\n" "$@" >> "$INSTALLED"; fi\n', encoding="utf-8")
        (stubs / name).chmod(0o755)
    result = subprocess.run(["bash", "-e", "-c", step["run"]], capture_output=True, text=True, check=False,
                            env={**os.environ, **env, "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
                                 "INSTALLED": str(installed)})
    assert result.returncode == 0, result.stdout + result.stderr

    packages = set(installed.read_text(encoding="utf-8").split())
    assert {"libgtk-3-0", "libnss3", "libgbm1", _ALSA[row["image"]]} <= packages, packages


def test_linux_builds_run_for_tag_builds_only():
    jobs = _workflow()["jobs"]
    builds = linux_builds(jobs)
    assert set(builds.values()) == DIAGNOSTIC_JOBS
    for target, name in builds.items():
        job, needs = jobs[name], admitted(["validate"])
        for dispatch, runs in (("tag", True), ("candidate", True), ("commit", False), ("channel", False)):
            assert gate(job["if"], DOWNLOADABLE_DISPATCHES[dispatch], needs) is runs, (name, dispatch)
        assert gate(job["if"], DRY_DISPATCH, needs)
        assert not gate(job["if"], {**DOWNLOADABLE_DISPATCHES["candidate"], "release-phase": "publish"}, needs)
        needs["validate"]["outputs"][target] = "false"
        assert not gate(job["if"], DOWNLOADABLE_DISPATCHES["tag"], needs)


@pytest.mark.parametrize("target", ["linux-x64", "linux-arm64"])
@pytest.mark.parametrize("feed", [True, False])
def test_linux_stage_archives_the_appimage_under_its_tag_only(tmp_path, r2_server, target, feed):  # health: allow F811 -- pytest injects the imported r2_server fixture by parameter name
    jobs = _workflow()["jobs"]
    step = _stage_step(jobs[linux_builds(jobs)[target]])
    assert gate(step["if"], DOWNLOADABLE_DISPATCHES["tag"], {}, job_if=False)
    assert gate(step["if"], DOWNLOADABLE_DISPATCHES["candidate"], {}, job_if=False)
    assert not gate(step["if"], DRY_DISPATCH, {}, job_if=False)

    arch = target.removeprefix("linux-")
    release = tmp_path / "apps/desktop/release"
    unpacked = release / f"linux-{arch}-unpacked/resources/agent-payload"
    unpacked.mkdir(parents=True)
    (unpacked / "manifest.json").write_text("{}")
    image = release / f"HermesBundled-{CANARY_TAG[1:]}-linux-{arch}.AppImage"
    image.write_bytes(b"AppImage transport fixture")
    (release / "builder-debug.yml").write_text("unstaged: true\n")
    staged = {image.name}
    if feed:
        name = "canary-linux.yml" if arch == "x64" else "canary-linux-arm64.yml"
        (release / name).write_text("version: fixture\n")
        staged.add(name)
    base = f"http://127.0.0.1:{r2_server.server_port}/hermes-releases"
    result = shell_step(tmp_path, r2_server, "", "", {
        "HERMES_PAYLOAD_TAG": CANARY_TAG, "HERMES_ARCHIVE_TAG": CANARY_TAG, "RELEASE_COMMIT": SHA,
        "TARGET": target, "CLOUDFLARE_R2_PUBLIC_URL": base, "CLOUDFLARE_R2_ACCOUNT_ID": "loopback",
        "CLOUDFLARE_R2_ACCESS_KEY_ID": "test-inert", "CLOUDFLARE_R2_SECRET_ACCESS_KEY": "test-inert",
        "CLOUDFLARE_R2_BUCKET": "hermes-releases",
    }, script=step["run"])
    assert result.returncode == 0, result.stdout + result.stderr

    prefix = f"releases/tag/{CANARY_TAG}/"
    receipt = json.loads(r2_server.store[f"{prefix}handoff-{target}.json"][0])
    assert (receipt["tag"], receipt["commit"], receipt["name"]) == (CANARY_TAG, SHA, target)
    assert {row["path"] for row in receipt["files"]} == staged
    assert set(r2_server.store) == {prefix + path for path in staged | {f"handoff-{target}.json"}}
