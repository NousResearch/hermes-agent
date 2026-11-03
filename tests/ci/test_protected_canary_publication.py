"""Execute the native legs' receipt staging for canary, stable and commit builds with local Git and
HTTP only. (Canary publication itself is stable-release.yml publish-canary: see
tests/scripts/test_release_channels.py and tests/scripts/test_canary_release_pipeline.py.)

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
from tests.ci.desktop_release_roles import native_builds, stage_step
from tests.ci.test_desktop_release_tag_admission import _git, _seed_repo
from tests.scripts.test_release_r2 import r2_server

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.platforms("posix")


def workflow_jobs():
    return hermes_yaml.safe_load((ROOT / ".github/workflows/desktop-bundled-release.yml").read_text())["jobs"]



def _version(tag: str) -> str:
    """The version a fixture tag's packages carry (empty for a tagless commit build)."""
    from hermes_cli.update_channel import release_version

    return release_version(tag) or ""
def stage_script(target, mode):
    """The tag/commit receipt stage of the native build leg for *target*."""
    jobs = workflow_jobs()
    return stage_step(jobs[native_builds(jobs)[(target, mode)]])["run"]


@pytest.fixture
def canary(tmp_path, r2_server, monkeypatch):
    _, clone = _seed_repo(tmp_path)
    commit = _git("rev-parse", "HEAD", cwd=clone)
    tag = "canary.20260913T001000Z-v0.1.2"
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
           "RELEASE_NEEDS": "{}"}

    def run(script, **overrides):
        return subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script], cwd=clone,
                              env={**env, **overrides}, capture_output=True, text=True, timeout=60, check=False)

    return clone, identity, env, run


def native_leg(root, identity, env, platform, arch):
    root.mkdir(parents=True, exist_ok=True)
    version = _version(env["RELEASE_TAG"])
    stamp = {
        "tag": env["RELEASE_TAG"], "commit": env["RELEASE_COMMIT"],
        "baseVersion": _version(env["RELEASE_TAG"]),
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
    assert metadata["version"] == (env["FIXTURE_WINDOWS_VERSION"] if platform == "win32" else _version(env["RELEASE_TAG"]))

