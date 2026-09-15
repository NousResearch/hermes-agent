"""Tagless packaging enters the real builder with an exact source identity."""
from __future__ import annotations

import os
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.bundles import desktop

from tests.ci.test_desktop_release_tag_admission import _BASH, _child_env, _git, _seed_repo


@pytest.mark.parametrize("variant", ["bundled", "light"])
def test_desktop_build_reaches_the_managed_payload_with_commit_ref(tmp_path, monkeypatch, variant):
    from scripts.bundles import desktop_prepare
    from scripts.bundles.desktop_inputs import build_environment
    from scripts.bundles.desktop_prepare import BuildRequest, PreparedDesktop

    _, repo = _seed_repo(tmp_path)
    sha = _git("rev-parse", "HEAD", cwd=repo)
    monkeypatch.setenv("HERMES_PAYLOAD_TAG", "v9.9.9")
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setenv("BUILD_NUMBER", "123")
    defaults = {"HERMES_GUEST_ONBOARDING": "1", "HERMES_DATA_DIR_SUFFIX": "magic-test", "HERMES_HOME": None}
    monkeypatch.setenv("HERMES_BUNDLE_ENV_JSON", json.dumps(defaults))
    request = BuildRequest.create(repo, tag=None, commit=sha, variant=variant,
                                  work=tmp_path / "work", cache=tmp_path / "cache", bundle_env=defaults)
    from pm.lock import Facts, Lockfile
    from pm.registry import get_package
    from pm.store import tree_digest
    (repo / "pm").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).resolve().parents[2] / "pm/lock.json", repo / "pm/lock.json")
    lock = Lockfile(repo / "pm/lock.json")
    store = request.cache / "tools"
    binaries = {}
    for name in ("python", "node", "npm"):
        package = get_package(name)
        version = lock.version(name)
        assert version is not None
        entry = store / package.store_entry(version, request.target)
        binary = package.binary(entry, request.target)
        assert binary is not None
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"published fixture")
        Facts(store / "facts.json").record(name, version, entry.name, package.env(entry, request.target), store,
                                           target=request.target, artifacts=[a["sha256"] for a in lock.artifacts(name, request.target)],
                                           digest=tree_digest(entry))
        binaries[name] = binary
    prepared = PreparedDesktop(request, binaries["python"], binaries["node"], Path(sys.executable),
                               tmp_path / "native", tmp_path / "packager", None, {}, "fixture-toolchain")
    env = build_environment(prepared, variant, os.environ)
    assert env.get("HERMES_PAYLOAD_TAG", "") == ""
    assert env["HERMES_BUILD_COMMIT"] == sha
    assert env["GITHUB_SHA"] == sha
    assert env["HERMES_PYTHON"] == str(binaries["python"])
    assert env["HERMES_PAYLOAD_VERSION"] == "0.1.2"
    assert json.loads(env["HERMES_BUNDLE_ENV_JSON"]) == defaults
    assert "BUILD_NUMBER" not in env
    shutil.rmtree(repo / "pm")

    # Composition crosses the preparation seam exactly once; native/Node build
    # behavior is exercised through their real provider tests, not command spies.
    calls = []
    def prepare(value):
        calls.append(value)
        return tmp_path / "prepared.json"
    built = []
    monkeypatch.setattr(desktop_prepare, "prepare", prepare)
    monkeypatch.setattr(desktop, "build_prepared", lambda path, args: built.append((path, args)))
    desktop.build(repo, None, variant, ['--publish=never'], commit_build=sha)
    assert len(calls) == 1 and calls[0].commit == sha and calls[0].variant == variant
    assert built == [(tmp_path / "prepared.json", ['--publish=never'])]
    before = len(calls)
    for tag, commit in [('v0.1.2', sha), (None, 'b' * 40), (None, 'short')]:
        with pytest.raises(ValueError):
            desktop.build(repo, tag, variant, [], commit_build=commit)
        assert len(calls) == before



@pytest.mark.parametrize("tag,commit", [(None, "a" * 40), ("v1.2.3-canary.20260911120000", None)])
def test_store_build_rejects_nonstable_before_reading_or_preparing_repo(tmp_path, tag, commit):
    absent = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="Store.*stable"):
        desktop.build(absent, tag, "store", [], commit_build=commit)
    assert not absent.exists()


def test_termux_commit_args_reach_prerequisite_checks_without_mutation(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    out = tmp_path / "must-not-be-written"
    helper = tmp_path / "bin"
    helper.mkdir()
    # Empty prerequisite commands are not build substitutes: stop at the first
    # actual prerequisite check, before any payload or output is created.
    env = _child_env(PATH=str(helper), HERMES_PAYLOAD_TAG="")
    scripts = [repo / "scripts/termux/termux_build.sh", repo / "scripts/termux/build_deb.sh"]
    for script in scripts:
        args = ["--repo", str(repo), "--commit", "a" * 40, "--out", str(out)]
        if script.name == "build_deb.sh":
            args += ["--payload", str(tmp_path / "absent")]
        result = subprocess.run([_BASH, str(script), *args], env=env, cwd=tmp_path,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "usage:" not in result.stderr
        assert not out.exists()
