"""Prepared desktop inputs bind a source and reject stale files before building."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


def _project(tmp_path: Path) -> tuple[Path, str]:
    from pm.store import current_target
    source = tmp_path / "source with spaces"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="1.2.3"\n', encoding="utf-8")
    (source / "package.json").write_text('{"workspaces": ["apps/desktop"]}', encoding="utf-8")
    (source / "package-lock.json").write_text('{"packages": {}}', encoding="utf-8")
    (source / ".gitignore").write_text(".build/\n", encoding="utf-8")
    (source / "pm").mkdir()
    (source / "pm/lock.json").write_text(json.dumps({"schema": 1, "packages": {
        name: {"version": "1.0.0", "artifacts": {current_target(): {"url": "https://example.test/tool", "sha256": "a" * 64}}}
        for name in ("python", "node", "npm")
    }}), encoding="utf-8")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-m", "fixture"], cwd=source, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    return source, commit


def test_stable_build_accepts_the_admitted_commit_before_the_final_tag_exists(tmp_path, monkeypatch):
    from scripts.bundles.desktop_inputs import identity_environment
    from scripts.bundles.desktop_prepare import BuildRequest

    source, commit = _project(tmp_path)
    subprocess.run([
        "git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
        "-c", "tag.gpgSign=false", "tag", "-a", "rc.1-v1.2.4", "-m", "claim",
    ], cwd=source, check=True, env={**__import__("os").environ,
                                  "GIT_COMMITTER_DATE": "2026-08-29T01:02:03Z"})
    monkeypatch.setenv("RELEASE_CLAIM_TAG", "rc.1-v1.2.4")
    claim_object = subprocess.check_output(
        ["git", "rev-parse", "rc.1-v1.2.4"], cwd=source, text=True,
    ).strip()
    monkeypatch.setenv("RELEASE_CLAIM_OBJECT", claim_object)
    request = BuildRequest.create(
        source, tag="v1.2.4", commit=None, release_commit=commit, variant="bundled",
        work=tmp_path / "work", cache=tmp_path / "cache", bundle_env={},
    )

    assert request.commit == commit
    assert request.version == "1.2.4"
    # The payload identity stays plain; the attempt ref lives only in the claim env.
    assert request.tag == "v1.2.4"
    assert request.release_epoch == 1787965323
    env = identity_environment(request, "bundled", {"HERMES_RELEASE_EPOCH": "1"})
    assert env["HERMES_RELEASE_EPOCH"] == "1787965323"

    monkeypatch.setenv("RELEASE_CLAIM_OBJECT", "f" * 40)
    with pytest.raises(ValueError, match="exact claim tag object"):
        BuildRequest.create(
            source, tag="v1.2.4", commit=None, release_commit=commit, variant="bundled",
            work=tmp_path / "wrong-work", cache=tmp_path / "wrong-cache", bundle_env={},
        )


def test_stable_build_rejects_a_claim_tag_for_another_version(tmp_path, monkeypatch):
    from scripts.bundles.desktop_prepare import BuildRequest

    source, commit = _project(tmp_path)
    subprocess.run([
        "git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
        "-c", "tag.gpgSign=false", "tag", "-a", "rc.2-v1.2.5", "-m", "claim",
    ], cwd=source, check=True, env={**__import__("os").environ,
                                  "GIT_COMMITTER_DATE": "2026-08-29T01:02:03Z"})
    monkeypatch.setenv("RELEASE_CLAIM_TAG", "rc.2-v1.2.5")
    monkeypatch.setenv("RELEASE_CLAIM_OBJECT", subprocess.check_output(
        ["git", "rev-parse", "rc.2-v1.2.5"], cwd=source, text=True).strip())
    with pytest.raises(ValueError, match="exact claim tag"):
        BuildRequest.create(
            source, tag="v1.2.4", commit=None, release_commit=commit, variant="bundled",
            work=tmp_path / "work", cache=tmp_path / "cache", bundle_env={},
        )


def test_prepared_input_roundtrip_rejects_mutation_and_foreign_source(tmp_path):
    from scripts.bundles.desktop_prepare import BuildRequest, PreparedDesktop

    source, commit = _project(tmp_path)
    work = tmp_path / "work"
    cache = tmp_path / "cache"
    request = BuildRequest.create(source, tag=None, commit=commit, variant="light", work=work, cache=cache, bundle_env={})
    from pm.lock import Facts
    from pm.store import tree_digest
    from pm.registry import get_package
    store = cache / "tools"
    binaries = {}
    for name in ("python", "node", "npm"):
        package = get_package(name)
        entry = store / package.store_entry("1.0.0", request.target)
        binary = package.binary(entry, request.target)
        assert binary is not None
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"published tool fixture")
        binaries[name] = binary
        Facts(store / "facts.json").record(name, "1.0.0", entry.name, package.env(entry, request.target), store,
                                           target=request.target, artifacts=["a" * 64], digest=tree_digest(entry))
    work.mkdir()
    executable, node = binaries["python"], binaries["node"]
    icons = work / "icon-environment" / "bin" / "python"
    icons.parent.mkdir(parents=True)
    icons.write_bytes(b"prepared icon interpreter")
    icon_library = work / "icon-environment" / "library"
    icon_library.write_bytes(b"prepared library")
    native = work / "native"
    native.mkdir()
    (native / "binding.node").write_bytes(b"prepared native input")
    packager = work / "packager.json"
    packager.write_text("{}", encoding="utf-8")
    prepared = PreparedDesktop.record(request, python=executable, node=node, icon_python=icons,
                                      native=native, packager=packager, payload=None, native_toolchain="fixture-toolchain")
    path = work / "prepared.json"
    prepared.write(path)
    restored = PreparedDesktop.load(path)
    restored.validate()
    assert restored.request.commit == commit
    assert restored.request.version == "1.2.3"
    assert "environment" not in json.loads(path.read_text())
    icon_library.unlink()
    with pytest.raises(ValueError, match="changed|stale|missing"):
        restored.validate()
    icon_library.write_bytes(b"prepared library")
    (native / "binding.node").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed|stale"):
        restored.validate()
    (native / "binding.node").write_bytes(b"prepared native input")
    (source / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="9.9.9"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source|checkout"):
        restored.validate()


def test_prepared_paths_cannot_escape_their_owned_roots(tmp_path):
    from scripts.bundles.desktop_prepare import BuildRequest, PreparedDesktop

    source, commit = _project(tmp_path)
    work, cache = tmp_path / "work", tmp_path / "cache"
    request = BuildRequest.create(source, tag=None, commit=commit, variant="light", work=work, cache=cache, bundle_env={})
    external = tmp_path / "external"
    external.write_bytes(b"not preparation-owned")
    with pytest.raises(ValueError, match="outside|owned"):
        PreparedDesktop.record(request, python=external, node=external, icon_python=external,
                               native=external, packager=external, payload=None, native_toolchain="fixture-toolchain")


def test_prepared_cli_rejects_missing_result_without_provisioning(tmp_path):
    source = Path(__file__).resolve().parents[2]
    absent = tmp_path / "absent" / "prepared.json"
    result = subprocess.run([sys.executable, str(source / "scripts/bundles/desktop.py"), "--prepared", str(absent)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "preparation" in result.stderr.lower()
    assert "unrecognized arguments" not in result.stderr
    assert not absent.parent.exists()


def test_source_admission_rejects_dirty_checkout_and_invalid_store_before_writes(tmp_path):
    from scripts.bundles.desktop_prepare import BuildRequest

    source, commit = _project(tmp_path)
    work = tmp_path / "work"
    cache = tmp_path / "cache"
    for tag, selected in [(None, commit), ("v1.2.0+canary.20260911T120000Z", None)]:
        with pytest.raises(ValueError, match="Store.*stable"):
            BuildRequest.create(source, tag=tag, commit=selected, variant="store", work=work, cache=cache, bundle_env={})
    (source / "package.json").write_text("{}", encoding="utf-8")
    (source / "pyproject.toml").write_text('staged fixture contents', encoding="utf-8")
    subprocess.run(["git", "add", "pyproject.toml"], cwd=source, check=True)
    (source / "generated output").mkdir()
    (source / "generated output" / "receipt.json").write_text('untracked fixture contents', encoding="utf-8")
    with pytest.raises(ValueError, match="source|checkout") as rejected:
        BuildRequest.create(source, tag=None, commit=commit, variant="light", work=work, cache=cache, bundle_env={})
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=source, text=True,
    ).rstrip("\r\n")
    assert str(rejected.value).endswith("\n" + status)
    assert 'generated output/receipt.json' in str(rejected.value)
    assert 'fixture contents' not in str(rejected.value)
    assert not work.exists()
    assert not cache.exists()


def test_preparation_never_reuses_an_unowned_work_directory(tmp_path):
    from scripts.bundles.desktop_prepare import BuildRequest, prepare

    source, commit = _project(tmp_path)
    work = tmp_path / "someone-elses-files"
    work.mkdir()
    precious = work / "request.json"
    precious.write_text("do not replace", encoding="utf-8")
    request = BuildRequest.create(source, tag=None, commit=commit, variant="light",
                                  work=work, cache=tmp_path / "cache", bundle_env={})
    with pytest.raises(ValueError, match="owned"):
        prepare(request)
    assert precious.read_text() == "do not replace"


def test_checkout_lock_excludes_a_second_build_process(tmp_path):
    from scripts.bundles.desktop_inputs import build_lock

    source, _ = _project(tmp_path)
    project = Path(__file__).resolve().parents[2]
    probe = (
        "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
        "from scripts.bundles.desktop_inputs import build_lock; "
        "lock=build_lock(Path(sys.argv[2])); lock.__enter__(); print('acquired'); lock.__exit__(None,None,None)"
    )
    command = [sys.executable, "-I", "-c", probe, str(project), str(source)]
    with build_lock(source):
        blocked = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert blocked.returncode != 0 and "another desktop" in blocked.stderr
    released = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert released.returncode == 0, released.stderr
    assert released.stdout.strip() == "acquired"
