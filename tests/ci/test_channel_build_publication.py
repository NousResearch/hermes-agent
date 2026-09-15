"""Execute channel workflow seams and receipt transport; fixture bytes are not native qualification."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import hermes_yaml
import pytest

from hermes_cli.release_channels import canonical_json
from scripts.releases import channel_publish, handoff, r2
from scripts.releases.channels import preview_identity
from tests.ci.test_desktop_release_tag_admission import _seed_repo, _git
from tests.scripts.test_release_r2 import r2_server  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


def request_for(base):
    return {"schema": 1, "buildId": "a" * 32, "channel": "unknown-at-build-time", "sequence": 7,
            "repository": "fixture/repo", "commit": "b" * 40, "sourceVersion": "1.2.3",
            "version": "0.0.7", "windowsVersion": "0.0.7.0", "identity": preview_identity("unknown-at-build-time", "c" * 16),
            "bundleEnv": {"FEATURE": "different from the prior request"}, "publicBase": base}


@pytest.fixture
def staged_channel(tmp_path, r2_server, request, monkeypatch):
    from scripts.releases.r2_scope import R2Scope
    base = f"http://127.0.0.1:{r2_server.server_port}/hermes-releases"
    if getattr(request, "param", ""):
        monkeypatch.setenv("R2_DISPOSABLE_RUN", "98765-1" if request.param == "receiver" else request.param)
        monkeypatch.setenv("GITHUB_REPOSITORY_ID", "12345")
        monkeypatch.setenv("CLOUDFLARE_R2_PUBLIC_URL", base)
    scope = R2Scope.configured()
    receiver = getattr(request, "param", "") == "receiver"
    request = request_for(scope.public_base(base))
    if receiver:
        from scripts.releases.channel_releases import product_identity
        request.update(channel="stable", receiverCandidate=True, releaseTag="v0.0.7",
                       identity=product_identity("v0.0.7"), bundleEnv={})
    _origin, clone = _seed_repo(tmp_path)
    request.update(commit=_git("rev-parse", "HEAD", cwd=clone), sourceVersion="0.1.2")
    build = tmp_path / "build"
    build.mkdir()
    name = request["identity"]["artifactNamePascal"]
    for platform in ("darwin", "win32"):
        for arch in ("arm64", "x64"):
            native = "macos" if platform == "darwin" else "windows"
            metadata = {"platform": native, "arch": arch, "commit": request["commit"], "request": request,
                        "identity": request["identity"]["appId" if platform == "darwin" else "msixAppIdWithOrg"],
                        "version": request["version" if platform == "darwin" else "windowsVersion"]}
            if receiver:
                metadata["receiverProtocol"] = 1
            if platform == "darwin":
                metadata["teamId"] = "ABCDEFGHIJ"
                metadata["filename"] = f"{name}-0.0.7-mac-{arch}.zip"
                files = [f"{name}-0.0.7-mac-{arch}.{suffix}" for suffix in ("zip", "dmg", "zip.blockmap", "dmg.blockmap")]
                for file in files:
                    (build / file).write_bytes(f"native transport fixture, not signed: {file}".encode())
            else:
                metadata.update(publisher="CN=Fixture", applicationId=request["identity"]["appNamePascal"])
                files = [f"{name}-0.0.7-win-{arch}.msix"]
                stamp = {"commit": request["commit"], "source": "channel-build", "channelBuild": request}
                with zipfile.ZipFile(build / files[0], "w") as package:
                    package.writestr("AppxManifest.xml", '<Package><Identity Name="' + metadata["identity"]
                                     + f'" Publisher="CN=Fixture" Version="0.0.7.0" ProcessorArchitecture="{arch}"/>'
                                     + '<Applications><Application Id="' + metadata["applicationId"] + '"/></Applications></Package>')
                    package.writestr("app/resources/install-stamp.json", json.dumps(stamp))
            metadata_name = f"metadata-{native}-{arch}.json"
            (build / metadata_name).write_text(json.dumps(metadata), encoding="utf-8")
            handoff.stage_channel_build(request, f"{platform}-{arch}", build, [*files, metadata_name])
    bundle_name = f"{name}-0.0.7.0-win.msixbundle"
    with zipfile.ZipFile(build / bundle_name, "w") as archive:
        archive.writestr("AppxMetadata/AppxBundleManifest.xml", '<Bundle><Identity Name="'
                         + request["identity"]["msixAppIdWithOrg"] + '" Publisher="CN=Fixture" Version="0.0.7.0"/>'
                         + '<Packages><Package Type="application" Architecture="arm64"/>'
                         + '<Package Type="application" Architecture="x64"/></Packages></Bundle>')
    handoff.stage_channel_build(request, "windows-universal", build, [bundle_name])
    prefix = handoff.channel_prefix(request)
    r2_server.store[scope.key(prefix + "request.json")] = (canonical_json(request), '"request"')
    return request, build


def test_channel_handoff_binds_full_request_and_feed_bytes(tmp_path, r2_server, staged_channel):
    request, build = staged_channel
    digest = hashlib.sha256(canonical_json(request)).hexdigest()
    assert channel_publish.read_request(request["buildId"], digest, request["publicBase"], request["repository"]) == request
    with pytest.raises(ValueError, match="SHA256|digest"):
        channel_publish.read_request(request["buildId"], "f" * 64, request["publicBase"], request["repository"])
    fetched = tmp_path / "fetched"
    handoff.fetch_channel_build(request, list(channel_publish.NATIVE_LEGS), fetched, public_base=request["publicBase"])
    needs = {name: {"result": "success"} for name in channel_publish.REQUIRED_JOBS}
    manifest, feeds = channel_publish.assemble(request, fetched, needs=needs)
    assert {(p["platform"], p["arch"]) for p in manifest["packages"]} == {
        (platform, arch) for platform in ("darwin", "win32") for arch in ("arm64", "x64")}
    xml = ET.parse(next(p for p in feeds if p.suffix == ".appinstaller")).getroot()
    assert xml.find("{*}UpdateSettings") is None
    assert xml.find("{*}MainBundle").get("Name") == request["identity"]["msixAppIdWithOrg"]
    feed = hermes_yaml.safe_load(next(p for p in feeds if p.suffix == ".yml").read_text(encoding="utf-8"))
    assert feed["version"] == request["version"]
    for file in feed["files"]:
        filename = file["url"].rsplit("/", 1)[-1]
        assert file["size"] == (build / filename).stat().st_size
    changed = copy.deepcopy(request)
    changed["bundleEnv"]["FEATURE"] = "other packaging input"
    with pytest.raises(ValueError, match="identity"):
        handoff.fetch_channel_build(changed, ["win32-x64"], tmp_path / "wrong", public_base=request["publicBase"])
    assert not (tmp_path / "wrong").exists()
    for name in channel_publish.REQUIRED_JOBS:
        for result in ("failure", "skipped", "cancelled", None):
            bad = copy.deepcopy(needs)
            if result is None:
                del bad[name]
            else:
                bad[name]["result"] = result
            with pytest.raises(ValueError, match=name):
                channel_publish.assemble(request, fetched, needs=bad)
    receipt_path = fetched / "handoff-win32-x64.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    missing = {**receipt, "files": [row for row in receipt["files"] if not row["path"].startswith("metadata-")]}
    receipt_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="omits required"):
        channel_publish.assemble(request, fetched, needs=needs)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    metadata = fetched / "metadata-windows-x64.json"
    metadata.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt"):
        channel_publish.assemble(request, fetched, needs=needs)


@pytest.mark.parametrize("staged_channel", ["receiver"], indirect=True)
def test_scoped_receiver_publication_preserves_existing_head_and_requires_smoke(tmp_path, r2_server, staged_channel):
    from scripts.releases.channels import ChannelPublisher, R2ChannelStore
    from scripts.releases.r2_scope import R2Scope
    request, _ = staged_channel
    scope = R2Scope.configured()
    previous = {"buildId": "b" * 32, "sequence": 1,
                "manifestKey": "releases/channel-builds/" + "b" * 32 + "/build.json", "sha256": "e" * 64}
    record = {"schema": 1, "name": "stable", "repository": request["repository"], "policy": "stable-release",
              "state": "active", "revision": 1, "nextSequence": 8, "identity": request["identity"],
              "testOnly": True, "head": previous}
    key = scope.key("releases/channels/stable.json")
    r2_server.store[key] = (canonical_json(record), '"record"')
    publisher = ChannelPublisher(R2ChannelStore(*r2.credentials()), request["repository"], request["publicBase"],
                                 authorize=lambda *_: pytest.fail("Receiver must not enter production promotion"))
    before = dict(r2_server.store)
    with pytest.raises(ValueError, match="successful jobs"):
        channel_publish.publish_receiver(request, tmp_path / "rejected", needs={}, publisher=publisher)
    assert r2_server.store == before
    result = channel_publish.publish_receiver(request, tmp_path / "receiver", publisher=publisher,
        needs={job: {"result": "success"} for job in channel_publish.REQUIRED_JOBS})
    assert result["manifest"]["receiverProtocol"] == 1
    assert json.loads(publisher.reader.read_bytes(result["head"]["manifestKey"], result["head"]["sha256"])) == result["manifest"]
    assert json.loads(r2_server.store[key][0])["head"] == previous


def workflow_step(workflow, job, name):
    doc = hermes_yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8"))
    return next(step["run"] for step in doc["jobs"][job]["steps"] if step.get("name") == name)


def run_shell(tmp_path, r2_server, script, env, *, cwd=None):
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    driver = tools / "python-driver.py"
    driver.write_text('import runpy,sys\n' + f'sys.path.insert(0, {str(ROOT)!r})\n'
                      + 'from scripts.releases import r2\n'
                      + f'r2.s3_endpoint=lambda _: "http://127.0.0.1:{r2_server.server_port}"\n'
                      + 'args=sys.argv[1:]\n'
                      + 'if args[0]=="-m":\n    sys.argv=args[1:]\n    runpy.run_module(args[1],run_name="__main__")\n'
                      + 'elif args[0]=="-c":\n    sys.argv=args[1:]\n    exec(args[1])\n'
                      + 'elif args[0]=="-":\n    sys.argv=args\n    exec(compile(sys.stdin.read(),"workflow","exec"))\n'
                      + f'else:\n    sys.argv=args\n    runpy.run_path({str(ROOT)!r}+"/"+args[0],run_name="__main__")\n', encoding="utf-8")
    python = tools / "python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"\n', encoding="utf-8")
    python.chmod(0o755)
    gh = tools / "gh"
    gh.write_text('#!/bin/sh\n[ "$1" = api ] || exit 3\nprintf "write\\n"\n', encoding="utf-8")
    gh.chmod(0o755)
    return subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script], cwd=cwd or tmp_path,
                          env={**os.environ, **env, "PATH": str(tools) + os.pathsep + os.environ["PATH"]},
                          capture_output=True, text=True, encoding="utf-8", timeout=60)


@pytest.mark.platforms("posix")
def test_real_workflow_admission_and_public_smoke_fetch(tmp_path, r2_server, staged_channel):
    request, _ = staged_channel
    clone = tmp_path / "clone"
    prefix = handoff.channel_prefix(request)
    r2_server.store[prefix + "request.json"] = (canonical_json(request), '"request"')
    digest = hashlib.sha256(canonical_json(request)).hexdigest()
    env = {"CHANNEL_BUILD": request["buildId"], "CHANNEL_REQUEST_SHA256": digest, "CLOUDFLARE_R2_PUBLIC_URL": request["publicBase"],
           "DEFAULT_BRANCH": "main", "GITHUB_REPOSITORY": "fixture/repo", "GITHUB_REF": "refs/heads/main",
           "GITHUB_WORKFLOW_REF": "fixture/repo/.github/workflows/desktop-bundled-release.yml@refs/heads/main",
           "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_SHA": request["commit"], "GITHUB_ACTOR": "fixture",
           "GITHUB_TRIGGERING_ACTOR": "fixture", "GITHUB_OUTPUT": str(tmp_path / "outputs"),
           "TAG": "", "BUILD_COMMIT": "", "RELEASE_PHASE": "", "UPLOAD_RELEASE": "false", "BUNDLE_ENV_JSON": "{}"}
    script = workflow_step("desktop-bundled-release.yml", "validate", "Validate tag shape, pyproject lockstep, and ancestry on origin/main")
    result = run_shell(tmp_path, r2_server, script, env, cwd=clone)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"sha={request['commit']}" in (tmp_path / "outputs").read_text(encoding="utf-8")
    for overrides in ({"GITHUB_REF": "refs/heads/feature"}, {"TAG": "v0.1.2"}, {"TERMUX_ONLY": "true"},
                      {"CHANNEL_REQUEST_SHA256": "f" * 64}):
        failed = run_shell(tmp_path, r2_server, script, {**env, **overrides}, cwd=clone)
        assert failed.returncode != 0
    # Re-stage a real ZIP-format MSIX fixture bound to the new admitted commit.
    package_root = tmp_path / "package"
    package_root.mkdir()
    filename = request["identity"]["artifactNamePascal"] + "-0.0.7-win-x64.msix"
    (package_root / filename).write_bytes(b"inert HTTP download fixture, not a native smoke")
    r2_server.store.pop(prefix + "handoff-win32-x64.json")
    r2_server.store.pop(prefix + filename)
    handoff.stage_channel_build(request, "win32-x64", package_root, [filename])
    fetch = workflow_step("desktop-bundle-smoke.yml", "windows", "Fetch one exact downloadable artifact")
    smoke = tmp_path / "smoke"
    env.update(SMOKE_ROOT=str(smoke), PUBLIC_BASE=request["publicBase"], RELEASE_TAG="", RELEASE_COMMIT=request["commit"],
               COMMIT_BUILD="false", PLATFORM="win32", ARCH="x64", FORMAT="msix")
    result = run_shell(tmp_path, r2_server, fetch, env)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads((smoke / "out/download.json").read_text(encoding="utf-8"))
    assert evidence["request"] == request
    assert (smoke / "download" / filename).read_bytes() == (package_root / filename).read_bytes()
    r2_server.store[prefix + filename] = (b"corrupt", '"corrupt"')
    failed = run_shell(tmp_path, r2_server, fetch, env)
    assert failed.returncode != 0
    assert (smoke / "download" / filename).read_bytes() == (package_root / filename).read_bytes()


@pytest.mark.platforms("posix")
def test_workflow_promotion_missing_native_gate_does_not_write(tmp_path, r2_server, staged_channel):
    request, _ = staged_channel
    script = workflow_step("desktop-bundled-release.yml", "publish-channel", "Publish immutable feeds and manifest, then CAS channel head")
    env = {"CHANNEL_BUILD": request["buildId"], "CHANNEL_REQUEST_SHA256": hashlib.sha256(canonical_json(request)).hexdigest(),
           "CLOUDFLARE_R2_PUBLIC_URL": request["publicBase"], "GITHUB_REPOSITORY": request["repository"],
           "RUNNER_TEMP": str(tmp_path), "RELEASE_NEEDS": "{}"}
    before = dict(r2_server.store)
    result = run_shell(tmp_path, r2_server, script, env)
    assert result.returncode != 0 and "successful jobs" in result.stderr
    assert dict(r2_server.store) == before


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("staged_channel", ["", "98765-1"], indirect=True)
def test_real_publication_cas_and_manifest_summary(tmp_path, r2_server, staged_channel):
    request, _ = staged_channel
    from scripts.releases.r2_scope import R2Scope
    scope = R2Scope.configured()
    channel_key = scope.key(f"releases/channels/{request['channel']}.json")
    record = {"schema": 1, "name": request["channel"], "repository": request["repository"],
              "policy": "preview", "state": "active", "revision": 1, "nextSequence": 8,
              "identity": request["identity"], "head": None}
    r2_server.store[channel_key] = (canonical_json(record), '"record"')
    script = workflow_step("desktop-bundled-release.yml", "publish-channel", "Publish immutable feeds and manifest, then CAS channel head")
    env = {"CHANNEL_BUILD": request["buildId"], "CHANNEL_REQUEST_SHA256": hashlib.sha256(canonical_json(request)).hexdigest(),
           "CLOUDFLARE_R2_PUBLIC_URL": request["publicBase"], "GITHUB_REPOSITORY": request["repository"],
           "RUNNER_TEMP": str(tmp_path), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
           "RELEASE_NEEDS": json.dumps({name: {"result": "success"} for name in channel_publish.REQUIRED_JOBS}),
           "DEFAULT_BRANCH": "main", "GITHUB_REF": "refs/heads/main", "GITHUB_EVENT_NAME": "workflow_dispatch",
           "GITHUB_WORKFLOW_REF": "fixture/repo/.github/workflows/desktop-bundled-release.yml@refs/heads/main",
           "GITHUB_SHA": request["commit"], "GITHUB_ACTOR": "fixture", "GITHUB_TRIGGERING_ACTOR": "fixture"}
    prefix = scope.prefix + handoff.channel_prefix(request)
    if scope.prefix:
        env["CLOUDFLARE_R2_PUBLIC_URL"] = os.environ["CLOUDFLARE_R2_PUBLIC_URL"]
        env["GITHUB_ACTIONS"] = "true"
        # A production sentinel makes accidental writes outside scope observable.
        r2_server.store["releases/channels/stable.json"] = (b"production sentinel", '"production"')
    artifact_key = prefix + request["identity"]["artifactNamePascal"] + "-0.0.7-mac-x64.zip"
    saved = r2_server.store[artifact_key]
    r2_server.store[artifact_key] = (b"damaged artifact", '"damaged"')
    result = run_shell(tmp_path, r2_server, script, env, cwd=tmp_path / "clone")
    assert result.returncode != 0 and "checksum mismatch" in result.stderr
    assert json.loads(r2_server.store[channel_key][0])["head"] is None
    assert prefix + "build.json" not in r2_server.store
    r2_server.store[artifact_key] = saved
    result = run_shell(tmp_path, r2_server, script, env, cwd=tmp_path / "clone")
    assert result.returncode == 0, result.stdout + result.stderr
    stored = json.loads(r2_server.store[channel_key][0])
    manifest_key = scope.key(handoff.channel_prefix(request) + "build.json")
    raw = r2_server.store[manifest_key][0]
    assert stored["head"]["sha256"] == hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    assert manifest["request"] == request
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    for package in manifest["packages"]:
        assert request["publicBase"] + "/" + package["artifact"]["key"] in summary
    puts = [path for method, path, _ in r2_server.requests if method == "PUT"]
    assert puts[-1].endswith(channel_key)
    if scope.prefix:
        assert all(path.startswith("/hermes-releases/" + scope.prefix) for path in puts)
        assert r2_server.store["releases/channels/stable.json"][0] == b"production sentinel"
        smoke_env = {**env, "SMOKE_ROOT": str(tmp_path / "scoped-smoke"), "PUBLIC_BASE": request["publicBase"],
                     "RELEASE_TAG": "", "RELEASE_COMMIT": request["commit"], "COMMIT_BUILD": "false",
                     "PLATFORM": "win32", "ARCH": "x64", "FORMAT": "msix", "GITHUB_OUTPUT": str(tmp_path / "smoke-output")}
        fetch = workflow_step("desktop-bundle-smoke.yml", "windows", "Fetch one exact downloadable artifact")
        result = run_shell(tmp_path, r2_server, fetch, smoke_env)
        assert result.returncode == 0, result.stdout + result.stderr
        evidence = json.loads((tmp_path / "scoped-smoke/out/download.json").read_text())
        assert evidence["request"] == request
    # A retired publisher can stage immutable diagnostics, but never revive its pointer.
    retired = {**stored, "state": "retired", "destination": "stable", "minimumVersion": "1.2.3",
               "lastHead": stored["head"], "destinationHead": stored["head"], "receiverProtocol": 1,
               "receiver": {"kind": "discontinued"}}
    r2_server.store[channel_key] = (canonical_json(retired), '"retired"')
    result = run_shell(tmp_path, r2_server, script, env, cwd=tmp_path / "clone")
    assert result.returncode != 0 and "Retired" in result.stderr
    assert json.loads(r2_server.store[channel_key][0]) == retired


@pytest.mark.platforms("posix")
def test_channel_windows_record_stage_and_assembly_handoff_shell(tmp_path, r2_server, staged_channel):
    import shutil
    request, build = staged_channel
    release = tmp_path / "apps/desktop/release"
    release.mkdir(parents=True)
    filename = request["identity"]["artifactNamePascal"] + "-0.0.7-win-x64.msix"
    shutil.copy2(build / filename, release / filename)
    request_file = tmp_path / "channel-request.json"
    request_file.write_text(json.dumps(request), encoding="utf-8")
    env = {"GITHUB_WORKSPACE": str(tmp_path), "RUNNER_TEMP": str(tmp_path), "TARGET": "win32-x64",
           "RELEASE_COMMIT": request["commit"], "CHANNEL_BUILD": request["buildId"],
           "HERMES_PAYLOAD_TAG": "", "HERMES_BUILD_COMMIT": ""}
    # This test owns the first metadata publication for this leg; the setup's
    # transport metadata is deliberately not a native recorder output.
    prefix = handoff.channel_prefix(request)
    for key in ("metadata-windows-x64.json", "handoff-win32-x64.json"):
        r2_server.store.pop(prefix + key)
    script = workflow_step("desktop-bundled-release.yml", "build-win32-commit", "Record and stage channel windows packages")
    result = run_shell(tmp_path, r2_server, script, env)
    assert result.returncode == 0, result.stdout + result.stderr
    # One-dispatch: the env-default commit equals the request's own, which is
    # provenance, not selection — the channel stage must accept it.
    for key in ("metadata-windows-x64.json", "handoff-win32-x64.json"):
        r2_server.store.pop(prefix + key)
    result = run_shell(tmp_path, r2_server, script, dict(env, HERMES_BUILD_COMMIT=request["commit"]))
    assert result.returncode == 0, result.stdout + result.stderr
    # A different commit stays an override attempt (and stages nothing).
    result = run_shell(tmp_path, r2_server, script, dict(env, HERMES_BUILD_COMMIT="d" * 40))
    assert result.returncode != 0 and "another commit" in result.stderr
    receipt_key = handoff.channel_prefix(request) + "handoff-win32-x64.json"
    receipt = json.loads(r2_server.store[receipt_key][0])
    assert receipt["request"] == request
    metadata = json.loads((release / "metadata-windows-x64.json").read_text(encoding="utf-8"))
    assert metadata["identity"] == request["identity"]["msixAppIdWithOrg"]
    assert metadata["applicationId"] == request["identity"]["appNamePascal"]
    fetch = workflow_step("desktop-bundled-release.yml", "assemble-win32-bundle", "Retrieve Windows packages from R2")
    result = run_shell(tmp_path, r2_server, fetch, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in release.glob("*.msix")) == sorted(p.name for p in build.glob("*.msix"))
    # Native admission validates package stamps before the stage can claim completion.
    with zipfile.ZipFile(release / filename) as package:
        manifest = package.read('AppxManifest.xml')
    with zipfile.ZipFile(release / filename, "w") as package:
        package.writestr("AppxManifest.xml", manifest)
        package.writestr("app/resources/install-stamp.json", json.dumps({"source": "commit-build", "commit": request["commit"]}))
    before = dict(r2_server.store)
    result = run_shell(tmp_path, r2_server, script, env)
    assert result.returncode != 0 and "provenance" in result.stderr
    assert dict(r2_server.store) == before
    gate = workflow_step("desktop-bundle-smoke.yml", "validate", "Reject empty or unsupported native smoke jobs")
    for target, expected in (("darwin/arm64/dmg", 0), ("win32/x64/msixbundle", 0), ("//", 1), ("linux/x64/zip", 1)):
        result = run_shell(tmp_path, r2_server, gate, {"TARGET": target})
        assert result.returncode == expected


@pytest.mark.platforms("posix")
def test_disposable_controller_allocates_then_separate_admission(tmp_path):
    from tests.scripts.test_release_channels import object_server
    _origin, clone = _seed_repo(tmp_path)
    commit = _git("rev-parse", "HEAD", cwd=clone)
    with object_server() as (url, objects, headers, requests, faults):
        # run_shell redirects only the R2 network endpoint; the controller,
        # permission CLI, pushed Git source and both workflow scripts run for real.
        from types import SimpleNamespace
        server = SimpleNamespace(server_port=int(url.rsplit(":", 1)[1]))
        env = {"DISPOSABLE_CHANNEL": "native-preview", "BUILD_COMMIT": commit,
               "CLOUDFLARE_R2_BUCKET": "bucket", "CLOUDFLARE_R2_PUBLIC_URL": url + "/bucket",
               "CLOUDFLARE_R2_ACCOUNT_ID": "loopback", "CLOUDFLARE_R2_ACCESS_KEY_ID": "fixture",
               "CLOUDFLARE_R2_SECRET_ACCESS_KEY": "fixture", "GITHUB_REPOSITORY": "fixture/repo",
               "GITHUB_REPOSITORY_ID": "12345", "GITHUB_RUN_ID": "98765", "GITHUB_RUN_ATTEMPT": "1",
               "GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "workflow_dispatch",
               "GITHUB_REF": "refs/heads/main", "DEFAULT_BRANCH": "main", "GITHUB_SHA": commit,
               "GITHUB_WORKFLOW_REF": "fixture/repo/.github/workflows/desktop-bundled-release.yml@refs/heads/main",
               "GITHUB_ACTOR": "fixture", "GITHUB_TRIGGERING_ACTOR": "fixture", "UPLOAD_RELEASE": "false",
               "GITHUB_STEP_SUMMARY": str(tmp_path / "allocation.md"), "BUNDLE_ENV_JSON": "{}"}
        script = workflow_step("desktop-bundled-release.yml", "allocate-disposable", "Probe scoped storage and allocate without dispatching")
        objects["releases/channels/stable.json"] = b"production sentinel"
        result = run_shell(tmp_path, server, script, env, cwd=clone)
        assert result.returncode == 0, result.stdout + result.stderr
        allocation = json.loads(result.stdout)
        request = allocation["request"]
        assert request["publicBase"] == url + "/bucket/ci-disposable/12345/98765"
        assert allocation["storagePrefix"] == "ci-disposable/12345/98765/"
        assert "disposable_run=98765" in allocation["command"]
        assert "channel_build=" + request["buildId"] in allocation["command"]
        assert objects["releases/channels/stable.json"] == b"production sentinel"
        assert all(key.startswith(allocation["storagePrefix"]) for method, key in requests if method == "PUT")
        before = dict(objects)
        # One dispatch: a preset R2_DISPOSABLE_RUN no longer fails admission — the
        # allocator derives and overwrites the lease from GITHUB_RUN_ID.
        for change in ({"DISPOSABLE_CHANNEL": "stable"},
                       {"GITHUB_REF": "refs/heads/other"}, {"GITHUB_ACTOR": "", "GITHUB_TRIGGERING_ACTOR": ""}):
            failed = run_shell(tmp_path, server, script, {**env, **change}, cwd=clone)
            assert failed.returncode != 0
            assert objects == before
        admit_env = {**env, "DISPOSABLE_CHANNEL": "", "BUILD_COMMIT": "", "TAG": "",
                     "R2_DISPOSABLE_RUN": allocation["disposableRun"], "CHANNEL_BUILD": request["buildId"],
                     "CHANNEL_REQUEST_SHA256": allocation["requestSha256"], "GITHUB_OUTPUT": str(tmp_path / "admitted")}
        admission = workflow_step("desktop-bundled-release.yml", "validate", "Validate tag shape, pyproject lockstep, and ancestry on origin/main")
        result = run_shell(tmp_path, server, admission, admit_env, cwd=clone)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "public-base=" + request["publicBase"] in (tmp_path / "admitted").read_text()
        for change in ({"R2_DISPOSABLE_RUN": ""}, {"GITHUB_REPOSITORY_ID": ""},
                       {"CLOUDFLARE_R2_PUBLIC_URL": ""}, {"R2_DISPOSABLE_RUN": "../escape"},
                       {"R2_DISPOSABLE_RUN": "98765-2"}, {"CHANNEL_REQUEST_SHA256": "f" * 64}):
            failed = run_shell(tmp_path, server, admission, {**admit_env, **change}, cwd=clone)
            assert failed.returncode != 0
            assert objects == before


@pytest.mark.platforms("posix")
def test_receiver_allocation_uses_official_identity_only_inside_scope(tmp_path):
    from tests.scripts.test_release_channels import object_server, publisher
    from scripts.releases.channel_disposable import allocate_receivers
    from scripts.releases.channel_releases import product_identity
    from scripts.releases.r2_scope import R2Scope

    with object_server() as (url, objects, *_):
        pub = publisher(url)
        with pytest.raises(ValueError, match="disposable"):
            allocate_receivers(pub, "a" * 40, "1.2.3", "a" * 40)
        scope = R2Scope("ci-disposable/12345/17-1/")
        pub.store.scope = scope
        pub.public_base += "/" + scope.prefix.rstrip("/")
        from hermes_cli.release_channels import ChannelReader
        pub.reader = ChannelReader(pub.public_base, pub.repository)
        receivers = allocate_receivers(pub, "a" * 40, "1.2.3", "a" * 40)
        assert set(receivers) == {"S", "T"}
        for slot, request in receivers.items():
            assert request["receiverCandidate"] is True
            assert request["identity"] == product_identity("v" + request["version"])
            assert request["publicBase"] == pub.public_base
            assert request["controllerCommit"] == "a" * 40
        assert receivers["T"]["sequence"] > receivers["S"]["sequence"]
        assert pub._read("stable")[0]["head"] is None


def test_pinned_xml_retains_legacy_automatic_policy(tmp_path):
    from scripts.bundles.release_artifacts import write_appinstaller
    for policy in ("automatic", "pinned"):
        out = tmp_path / f"{policy}.appinstaller"
        write_appinstaller(out, identity="Fixture.App", publisher="CN=Fixture", version="0.0.7.0",
                           self_uri="https://archive.example/build/stable.appinstaller",
                           artifact_uri="https://archive.example/build/app.msixbundle", update_policy=policy)
        xml = ET.parse(out).getroot()
        assert (xml.find("{*}UpdateSettings/{*}OnLaunch") is not None) == (policy == "automatic")
        assert xml.find("{*}UpdateSettings/{*}ForceUpdateFromAnyVersion") is None
