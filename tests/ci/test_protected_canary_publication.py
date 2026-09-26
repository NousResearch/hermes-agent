"""Execute protected canary workflow/controller with local Git and HTTP only.

The packages and codesign/GitHub executables are transport fixtures, not native
qualification. No admission, metadata recorder or channel publisher is mocked.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import plistlib
import shlex
import subprocess
import sys
import zipfile

import hermes_yaml
import pytest

from hermes_cli.release_channels import ChannelReader
from scripts.releases import channel_releases, handoff
from tests.ci.desktop_release_roles import canary_publisher, native_builds, stage_step
from tests.ci.test_desktop_release_tag_admission import _git, _seed_repo
from tests.scripts.test_release_r2 import r2_server  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.platforms("posix")


def workflow_jobs():
    return hermes_yaml.safe_load((ROOT / ".github/workflows/desktop-bundled-release.yml").read_text())["jobs"]


def canary_job():
    jobs = workflow_jobs()
    return jobs[canary_publisher(jobs)]


def stage_script(target, mode):
    """The tag/commit receipt stage of the native build leg for *target*."""
    jobs = workflow_jobs()
    return stage_step(jobs[native_builds(jobs)[(target, mode)]])["run"]


def step_script(job, name):
    return next(step["run"] for step in job["steps"] if step.get("name") == name)


@pytest.fixture
def canary(tmp_path, r2_server, monkeypatch):
    _, clone = _seed_repo(tmp_path)
    commit = _git("rev-parse", "HEAD", cwd=clone)
    tag = "v0.1.2+canary.20260913T001000Z"
    _git("tag", "-a", tag, "-m", '{"schema":1}', cwd=clone)
    tag_object = _git("rev-parse", f"{tag}^{{tag}}", cwd=clone)
    _git("push", "origin", tag, cwd=clone)
    desktop = clone / "apps/desktop"
    desktop.mkdir(parents=True)
    (desktop / "product-identity.cjs").symlink_to(ROOT / "apps/desktop/product-identity.cjs")
    monkeypatch.chdir(clone)
    identity = channel_releases.product_identity(tag)
    windows_version = subprocess.check_output([
        "node", "--input-type=module", "-e",
        f"import {{nativeQuad}} from {json.dumps((ROOT / 'scripts/msix-shared.mjs').as_uri())};"
        f"console.log(nativeQuad({json.dumps(tag)}, Date.parse('2026-09-13T00:10:00Z') / 1000))",
    ], text=True).strip()
    assert windows_version == "26.913.0.1000"
    tools = tmp_path / "tools"
    tools.mkdir()
    driver = tools / "driver.py"
    driver.write_text(
        "import runpy,sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from scripts.releases import r2\n"
        f"r2.s3_endpoint=lambda _: 'http://127.0.0.1:{r2_server.server_port}'\n"
        "args=sys.argv[1:]\nsys.argv=args[1:]\n"
        "assert args[0]=='-m', args\nrunpy.run_module(args[1],run_name='__main__')\n"
    )
    python = tools / "python"
    python.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(driver))} \"$@\"\n")
    python.chmod(0o755)
    release_state = tmp_path / "github-release.json"
    release_state.write_text(json.dumps({"tagName": tag, "isDraft": False, "isPrerelease": True}))
    gh = tools / "gh"
    gh.write_text(
        f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\nstate=Path(os.environ['FIXTURE_RELEASE'])\n"
        "if args[:1]==['api']: print('main')\n"
        "elif args[:2]==['release','view']: print(state.read_text())\n"
        "elif args[:2]==['release','edit']:\n"
        " data=json.loads(state.read_text());data['isDraft']=False;state.write_text(json.dumps(data))\n"
        "else: raise SystemExit('unexpected gh call: '+repr(args))\n"
    )
    gh.chmod(0o755)
    # Only the signing command is a fixture boundary; real macOS CI owns its proof.
    codesign = tools / "codesign"
    codesign.write_text('#!/bin/sh\nprintf "TeamIdentifier=ABCDEFGHIJ\\n" >&2\n')
    codesign.chmod(0o755)
    base = f"http://127.0.0.1:{r2_server.server_port}/hermes-releases"
    env = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
           "FIXTURE_RELEASE": str(release_state), "FIXTURE_WINDOWS_VERSION": windows_version, "GITHUB_ACTIONS": "true",
           "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REPOSITORY": "NousResearch/hermes-agent",
           "GITHUB_WORKFLOW_REF": "NousResearch/hermes-agent/.github/workflows/desktop-bundled-release.yml@refs/heads/main",
           "RELEASE_TAG": tag, "TAG": tag, "HERMES_PAYLOAD_TAG": tag,
           "RELEASE_TAG_OBJECT": tag_object,
           "RELEASE_COMMIT": commit, "RELEASE_PHASE": "", "HERMES_DESKTOP_VARIANT": "bundled",
           "HERMES_BUILD_COMMIT": "", "CHANNEL_BUILD": "", "R2_DISPOSABLE_RUN": "",
           "CLOUDFLARE_R2_PUBLIC_URL": base,
           "RELEASE_NEEDS": json.dumps({name: {"result": "success"} for name in canary_job()["needs"]})}

    def run(script, **overrides):
        return subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script], cwd=clone,
                              env={**env, **overrides}, capture_output=True, text=True, timeout=60)

    return clone, identity, env, run


def native_leg(root, identity, env, platform, arch):
    root.mkdir(parents=True, exist_ok=True)
    version = env["RELEASE_TAG"][1:]
    stamp = {
        "tag": env["RELEASE_TAG"], "commit": env["RELEASE_COMMIT"],
        "baseVersion": env["RELEASE_TAG"][1:],
    }
    artifact = identity["artifactNamePascal"]
    if platform == "win32":
        (root / f"version-info-{arch}.json").write_text(json.dumps({
            "productVersion": env["FIXTURE_WINDOWS_VERSION"],
        }))
        with zipfile.ZipFile(root / f"{artifact}-{version}-win-{arch}.msix", "w") as package:
            package.writestr("AppxManifest.xml", f'<Package><Identity Name="{identity["msixAppIdWithOrg"]}" Publisher="CN=Fixture" Version="{env["FIXTURE_WINDOWS_VERSION"]}" ProcessorArchitecture="{arch}"/><Applications><Application Id="{identity["appNamePascal"]}"/><Application Id="CLI"><VisualElements AppListEntry="none"/></Application></Applications></Package>')
            package.writestr("app/resources/install-stamp.json", json.dumps(stamp))
    else:
        name = f"{artifact}-{version}-mac-{arch}"
        (root / f"mac-{arch}/{artifact}.app").mkdir(parents=True)
        with zipfile.ZipFile(root / f"{name}.zip", "w") as package:
            package.writestr(f"{artifact}.app/Contents/Info.plist", plistlib.dumps({"CFBundleIdentifier": identity["appId"], "CFBundleShortVersionString": version}))
            package.writestr(f"{artifact}.app/Contents/Resources/install-stamp.json", json.dumps(stamp))
        (root / "canary-mac.yml").write_text(json.dumps({"version": version}))
        for suffix in ("dmg", "dmg.blockmap", "zip.blockmap"):
            (root / f"{name}.{suffix}").write_bytes(b"unsigned transport fixture")


@pytest.mark.parametrize("platform", ["win32", "darwin"])
@pytest.mark.parametrize("variant,phase,commit_build", [
    ("bundled", "", False), ("bundled", "candidate", False),
    ("light", "", False), ("bundled", "", True),
])
def test_canary_metadata_is_recorded_and_receipt_bound(canary, r2_server, platform, variant, phase, commit_build):
    clone, identity, original_env, run = canary
    env = {**original_env, "HERMES_DESKTOP_VARIANT": variant, "RELEASE_PHASE": phase}
    if phase == "candidate":
        env["RELEASE_TAG"] = env["HERMES_PAYLOAD_TAG"] = "v0.1.2"
        env["FIXTURE_WINDOWS_VERSION"] = "0.1.2.0"
    if commit_build:
        env["RELEASE_TAG"] = env["HERMES_PAYLOAD_TAG"] = ""
        env["HERMES_BUILD_COMMIT"] = env["RELEASE_COMMIT"]
    root = clone / "apps/desktop/release"
    native_leg(root, identity, env, platform, "x64")
    script = stage_script(f"{platform}-x64", "commit" if commit_build else "release")
    result = run(script, **env, TARGET=f"{platform}-x64")
    assert result.returncode == 0, result.stdout + result.stderr
    prefix = f"releases/commit/{env['RELEASE_COMMIT']}/" if commit_build else f"releases/tag/{env['RELEASE_TAG']}/"
    receipt = json.loads(r2_server.store[prefix + handoff.receipt_name(f"{platform}-x64")][0])
    metadata_name = f"metadata-{'windows' if platform == 'win32' else 'macos'}-x64.json"
    if variant == "light" or commit_build:
        assert not any(row["path"].startswith("metadata-") for row in receipt["files"])
        return
    entry = next(row for row in receipt["files"] if row["path"] == metadata_name)
    body = r2_server.store[prefix + metadata_name][0]
    metadata = json.loads(body)
    assert entry["sha256"] == hashlib.sha256(body).hexdigest()
    assert metadata["commit"] == env["RELEASE_COMMIT"] and metadata["tag"] == env["RELEASE_TAG"]
    assert metadata["version"] == (env["FIXTURE_WINDOWS_VERSION"] if platform == "win32" else env["RELEASE_TAG"][1:])


def stage_canary(clone, identity, env, run):
    root = clone / "apps/desktop/release"
    # Stage independent per-architecture job workspaces through the actual shell.
    import shutil
    for platform in ("darwin", "win32"):
        for arch in ("arm64", "x64"):
            if root.exists():
                shutil.rmtree(root)
            native_leg(root, identity, env, platform, arch)
            result = run(stage_script(f"{platform}-{arch}", "release"), **env, TARGET=f"{platform}-{arch}")
            assert result.returncode == 0, result.stdout + result.stderr
    bundle = root / f"{identity['artifactNamePascal']}-{env['FIXTURE_WINDOWS_VERSION']}-win.msixbundle"
    with zipfile.ZipFile(bundle, "w") as package:
        package.writestr("AppxMetadata/AppxBundleManifest.xml", f'<Bundle><Identity Name="{identity["msixAppIdWithOrg"]}" Publisher="CN=Fixture" Version="{env["FIXTURE_WINDOWS_VERSION"]}"/><Packages><Package Type="application" Architecture="arm64"/><Package Type="application" Architecture="x64"/></Packages></Bundle>')
    result = run('python -m scripts.releases.handoff stage --tag "$RELEASE_TAG" --commit "$RELEASE_COMMIT" --name windows-universal --root apps/desktop/release --include "*.msixbundle"', **env)
    assert result.returncode == 0, result.stdout + result.stderr
    return bundle


def test_published_canary_workflow_advances_only_after_every_gate(canary, r2_server, monkeypatch, tmp_path):
    clone, identity, env, run = canary
    bundle = stage_canary(clone, identity, env, run)
    publisher = canary_job()
    script = step_script(publisher, "Publish the admitted canary and advance its protected head")
    before = dict(r2_server.store)
    for job in publisher["needs"]:
        for outcome in ("failure", "skipped", "cancelled", None):
            needs = json.loads(env["RELEASE_NEEDS"])
            if outcome is None:
                del needs[job]
            else:
                needs[job] = {"result": outcome}
            result = run(script, RELEASE_NEEDS=json.dumps(needs))
            assert result.returncode != 0 and job in result.stderr
            assert r2_server.store == before
    release = Path(env["FIXTURE_RELEASE"])
    published = json.loads(release.read_text())
    for bad in ({**published, "isPrerelease": False},):
        release.write_text(json.dumps(bad))
        result = run(script)
        assert result.returncode != 0 and "prerelease" in result.stderr
        assert r2_server.store == before
    release.write_text(json.dumps({**published, "isDraft": True}))
    result = run(script, RELEASE_TAG_OBJECT="b" * 40)
    assert result.returncode != 0 and "moved" in result.stderr
    assert json.loads(release.read_text())["isDraft"] is True
    assert r2_server.store == before
    release.write_text(json.dumps(published))
    prefix = f"releases/tag/{env['RELEASE_TAG']}/"
    for key in (prefix + "metadata-windows-arm64.json", prefix + bundle.name):
        saved = r2_server.store.pop(key)
        missing = dict(r2_server.store)
        result = run(script)
        assert result.returncode != 0
        assert r2_server.store == missing
        r2_server.store[key] = (b"corrupt", '"corrupt"')
        corrupted = dict(r2_server.store)
        result = run(script)
        assert result.returncode != 0 and "checksum mismatch" in result.stderr
        assert r2_server.store == corrupted
        r2_server.store[key] = saved
    assert r2_server.store == before
    # The controller owns the draft flip and its independent custody read-back.
    release.write_text(json.dumps({**published, "isDraft": True}))
    for step in publisher["steps"]:
        if "run" in step:
            result = run(step["run"])
            assert result.returncode == 0, result.stdout + result.stderr
    resolved = ChannelReader(env["CLOUDFLARE_R2_PUBLIC_URL"], repository=env["GITHUB_REPOSITORY"]).resolve("canary")
    assert resolved.manifest is not None
    assert resolved.manifest["request"]["windowsVersion"] == "26.913.0.1000"
    assert resolved.manifest["request"]["commit"] == env["RELEASE_COMMIT"]
    assert len(resolved.manifest["packages"]) == 4
    # Bootstrap reads actual receipt-bound artifacts and the promoted native feeds,
    # rather than asking canary for a stable-only candidate manifest.
    from scripts.bundles.release_artifacts import write_appinstaller
    from scripts.releases import stable
    from hermes_cli.release_channels import canonical_json
    mac = resolved.manifest["packages"][0]
    mac_feed = json.loads(r2_server.store[mac["feed"]["key"]][0])
    for entry in mac_feed["files"]:
        entry["url"] = entry["url"].removeprefix(env["CLOUDFLARE_R2_PUBLIC_URL"])
    mac_feed["path"] = mac_feed["path"].removeprefix(env["CLOUDFLARE_R2_PUBLIC_URL"])
    r2_server.store["releases/darwin/canary/canary-mac.yml"] = (canonical_json(mac_feed), '"feed"')
    win = next(p for p in resolved.manifest["packages"] if p["platform"] == "win32")
    promoted_key = "releases/win32/canary/" + bundle.name
    r2_server.store[promoted_key] = r2_server.store[win["artifact"]["key"]]
    descriptor = tmp_path / "canary.appinstaller"
    write_appinstaller(descriptor, identity=win["identity"], publisher=win["publisher"], version=win["version"],
                       self_uri=env["CLOUDFLARE_R2_PUBLIC_URL"] + "/releases/win32/canary/canary.appinstaller",
                       artifact_uri=env["CLOUDFLARE_R2_PUBLIC_URL"] + "/" + promoted_key)
    r2_server.store["releases/win32/canary/canary.appinstaller"] = (descriptor.read_bytes(), '"feed"')
    monkeypatch.setattr(stable, "output", lambda args: env["RELEASE_COMMIT"] if "/commits/" in args[2]
                        else json.dumps({"draft": False, "prerelease": True, "published_at": "fixture-published"}))
    assert channel_releases.verify_bootstrap(resolved.manifest["request"], resolved.manifest,
                                            env["CLOUDFLARE_R2_PUBLIC_URL"], env["GITHUB_REPOSITORY"])
    saved_feed = r2_server.store["releases/darwin/canary/canary-mac.yml"]
    bad = json.loads(saved_feed[0]); bad["version"] = "9.0.0"
    r2_server.store["releases/darwin/canary/canary-mac.yml"] = (canonical_json(bad), '"bad"')
    with pytest.raises(ValueError, match="promoted"):
        channel_releases.verify_bootstrap(resolved.manifest["request"], resolved.manifest,
                                         env["CLOUDFLARE_R2_PUBLIC_URL"], env["GITHUB_REPOSITORY"])
    r2_server.store["releases/darwin/canary/canary-mac.yml"] = saved_feed
    after = dict(r2_server.store)
    assert run(script).returncode == 0
    assert r2_server.store == after
    # A later canary really advances the existing head and native quad, rather
    # than merely succeeding at the empty-channel bootstrap case.
    newer_tag = "v0.1.2+canary.20260913T001100Z"
    _git("tag", "-a", newer_tag, "-m", '{"schema":1}', cwd=clone)
    newer_object = _git("rev-parse", f"{newer_tag}^{{tag}}", cwd=clone)
    _git("push", "origin", newer_tag, cwd=clone)
    newer = {**env, "RELEASE_TAG": newer_tag, "HERMES_PAYLOAD_TAG": newer_tag,
             "RELEASE_TAG_OBJECT": newer_object,
             "FIXTURE_WINDOWS_VERSION": "26.913.0.1100"}
    stage_canary(clone, identity, newer, run)
    release.write_text(json.dumps({**published, "tagName": newer_tag}))
    result = run(script, **newer)
    assert result.returncode == 0, result.stdout + result.stderr
    advanced = ChannelReader(env["CLOUDFLARE_R2_PUBLIC_URL"], repository=env["GITHUB_REPOSITORY"]).resolve("canary")
    assert advanced.manifest is not None
    assert advanced.manifest["request"]["windowsVersion"] == "26.913.0.1100"
    assert advanced.terminal["head"]["sequence"] > resolved.terminal["head"]["sequence"]
    after = dict(r2_server.store)
    release.write_text(json.dumps(published))
    result = run(script)
    assert result.returncode != 0 and "stale" in result.stderr
    assert r2_server.store == after
    # A moved published tag must not retarget the accepted release, even on retry.
    _git("commit", "--allow-empty", "-m", "new source", cwd=clone)
    _git("tag", "-f", "-a", env["RELEASE_TAG"], "-m", '{"schema":1}', cwd=clone)
    _git("push", "--force", "origin", env["RELEASE_TAG"], cwd=clone)
    result = run(script)
    assert result.returncode != 0 and "moved" in result.stderr
    assert r2_server.store == after
