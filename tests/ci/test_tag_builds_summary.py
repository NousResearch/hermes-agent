"""Run the tagged-build summary step against a disposable R2 transport."""
import json
import shlex
import sys

import pytest

from scripts.releases import r2
from tests.ci.desktop_release_roles import (
    CANARY_TAG, DOWNLOADABLE_DISPATCHES, admitted, gate, needs_of, selection_gates, tag_summary, termux_builder,
    updater_publishers,
)
from tests.ci.test_commit_build_staging import shell_step
from tests.ci.test_desktop_release_tag_admission import _workflow
from tests.scripts.test_release_r2 import r2_server  # noqa: F401


@pytest.mark.parametrize("gh_available", [False, True])
def test_admitted_failure_publishes_tag_info_without_promoting_channel(tmp_path, r2_server, gh_available):
    jobs = _workflow()["jobs"]
    table = tag_summary(jobs)
    job = jobs[table]
    gates = selection_gates(jobs)
    windows_publisher = updater_publishers(jobs)["win32"]
    termux = termux_builder(jobs)
    # This signing-context observer must survive failed needs without admitting
    # rejected tags, commit builds, dry runs, or stable release phases.
    tag = DOWNLOADABLE_DISPATCHES["tag"]

    def observes(inputs, **validate):
        needs = admitted(needs_of(job))
        needs["validate"]["outputs"]["all-jobs"] = "true"
        needs["validate"].update(validate)
        for name in needs:
            if name != "validate":
                needs[name]["result"] = "failure"
        return gate(job["if"], inputs, needs)

    assert observes(tag)
    assert not observes(tag, result="failure")
    assert not observes(tag, outputs={"all-jobs": "true", "sha": ""})
    for rejected in ({"build_commit": "a" * 40}, {"upload_release": False}, {"release-phase": "candidate"},
                     {"release-phase": "publish"}, {"channel": "preview"}):
        assert not observes({**tag, **rejected}), rejected
    render = next(step for step in job["steps"] if step.get("name") == "Render")
    assert render["env"]["RELEASE_NEEDS"] == "${{ toJSON(needs) }}"

    tag = CANARY_TAG
    run_url = "https://github.example/o/r/actions/runs/12345"
    base = f"http://127.0.0.1:{r2_server.server_port}/hermes-releases"
    channel_key = "releases/canary/index.html"
    previous = b'<meta name="hermes-build" content="v0.27.0+canary.20260817T101010Z">'
    r2_server.store[channel_key] = (previous, "text/html")
    helper = tmp_path / "bin"
    helper.mkdir()
    gh = helper / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"sys.exit(1) if not {gh_available!r} else None\n"
        "if sys.argv[1:3] == ['release', 'view']:\n"
        "    print(json.dumps({'body': '<!-- HERMES_BUILDS_TABLE -->'}))\n"
        "elif sys.argv[1:3] == ['release', 'edit']:\n"
        "    assert 'Build incomplete' in sys.stdin.read()\n"
        "else: raise AssertionError(sys.argv)\n",
        encoding="utf-8",
    )
    # Use a shell trampoline for interpreters whose full Nix path exceeds
    # the host's shebang limit.
    driver = helper / "gh-driver.py"
    gh.rename(driver)
    gh.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"\n')
    gh.chmod(0o755)
    needs = {name: {"result": "success"} for name in job["needs"]}
    for name in [*gates.values(), termux]:
        needs[name]["result"] = "failure"
    needs[windows_publisher]["result"] = "skipped"
    result = shell_step(tmp_path, r2_server, table, "Render", {
        "HERMES_PAYLOAD_TAG": tag, "GITHUB_REPOSITORY": "o/r",
        "RUN_URL": run_url,
        "RELEASE_NEEDS": json.dumps(needs), "CLOUDFLARE_R2_PUBLIC_URL": base,
        "CLOUDFLARE_R2_ACCOUNT_ID": "loopback", "CLOUDFLARE_R2_ACCESS_KEY_ID": "test-inert",
        "CLOUDFLARE_R2_SECRET_ACCESS_KEY": "test-inert", "CLOUDFLARE_R2_BUCKET": "hermes-releases",
    })
    assert (result.returncode == 0) is gh_available, result.stdout + result.stderr
    key = f"releases/tag/{tag}/index.html"
    assert key in r2_server.store, result.stdout + result.stderr
    page = r2_server.store[key][0].decode()
    release_url = r2.public_url_for("https://github.com/o/r/releases/tag", tag)
    assert f'href="{release_url}"' in page
    assert "Build incomplete" in page
    assert f"{gates['win32-x64']} (failure)" in page and f"{windows_publisher} (skipped)" in page
    assert "No downloadable artifacts" in page
    for name, info in needs.items():
        if info["result"] != "success":
            assert f'<td>{name} ({info["result"]})</td><td><a href="{run_url}">View build run</a>' in page
    assert base not in page
    assert render["env"]["RUN_URL"] == "${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
    assert r2.public_url_for(base, key) in result.stdout
    assert r2_server.store[channel_key][0] == previous
    assert set(r2_server.store) == {channel_key, key}

    # publish-canary depends on this job's success; rendering diagnostics must
    # not turn a failed Termux prerequisite into permission to publish a draft.
    needs = {name: {"result": "success"} for name in job["needs"]}
    for state in ("failure", "skipped", "success"):
        needs[termux]["result"] = state
        checked = shell_step(tmp_path, r2_server, table, "Preserve release success gate", {
            "RELEASE_NEEDS": json.dumps(needs),
        })
        assert (checked.returncode == 0) is (state == "success"), checked.stdout + checked.stderr
