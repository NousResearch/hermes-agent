"""The cache transport describes candidates without bootstrapping their owners."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.ci.setup_toolchain import current_target
from tests.ci.desktop_release_roles import gate


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci/desktop_build_cache.py"


def source_tree(root):
    root.mkdir()
    packages = {"": {}, "apps/editor": {"name": "editor"}, "ui/packages/helper": {"name": "helper"}}
    packages["node_modules/editor"] = {"link": True, "resolved": "apps/editor"}
    packages["node_modules/helper"] = {"link": True, "resolved": "ui/packages/helper"}
    for package in ("", "apps/editor", "ui/packages/helper"):
        directory = root / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "package.json").write_text(json.dumps({"name": package or "root"}), encoding="utf-8")
    (root / "package-lock.json").write_text(json.dumps({"lockfileVersion": 3, "packages": packages}), encoding="utf-8")
    for name in ("pyproject.toml", "uv.lock", "pm/lock.json", "pm/pyproject.toml", "pm/uv.lock"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    return root


def describe(source, cache, tmp_path, *extra):
    output = tmp_path / "github-output"
    output.unlink(missing_ok=True)
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "describe", "--source", str(source),
         "--cache", str(cache), "--producer", "desktop-production", *extra],
        cwd=tmp_path, env={**os.environ, "GITHUB_OUTPUT": str(output),
                           "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2", "GITHUB_JOB": "desktop"},
        capture_output=True, text=True, encoding="utf-8", timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout), dict(line.split("=", 1) for line in output.read_text(encoding="utf-8-sig").splitlines())


def test_stdlib_description_has_stable_allowlisted_paths_before_bootstrap(tmp_path, monkeypatch):
    source = source_tree(tmp_path / "source with spaces")
    cache = tmp_path / "reusable inputs"
    home = tmp_path / "private-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    description, outputs = describe(source, cache, tmp_path)

    expected = {str(cache / domain) for domain in (
        "tools", "python/runtime", "python/build", "npm/_cacache", "native", "packager",
    )}
    assert set(description["paths"]) == expected
    assert json.loads(outputs["cache-paths"]) == description["paths"]
    assert outputs["cache-key"] == description["key"]
    assert outputs["restore-prefix"] == description["restore_keys"][-1]
    assert outputs["input-prefix"] == description["restore_keys"][0]
    assert description["target"] == current_target()
    assert not cache.exists()
    assert not home.exists()

    for path in expected:
        Path(path).mkdir(parents=True, exist_ok=True)
    warm, _ = describe(source, cache, tmp_path)
    assert warm == description, "Actions path versions cannot depend on cache existence"


def test_keys_follow_dependency_inputs_not_checkout_or_release_identity(tmp_path, monkeypatch):
    from scripts.ci.desktop_build_cache import describe_cache

    source = source_tree(tmp_path / "source")
    cache = tmp_path / "cache"
    first = describe_cache(source, cache, "desktop-production")
    (source / "frontend.ts").write_text("new UI", encoding="utf-8")
    monkeypatch.setenv("GITHUB_SHA", "new-commit")
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v9.0.0")
    assert describe_cache(source, cache, "desktop-production") == first

    for relative in ("apps/editor/package.json", "package-lock.json", "pm/lock.json", "uv.lock",
                     "scripts/build/node-deps.mjs", "apps/desktop/scripts/stage-native-deps.mjs",
                     "scripts/bundles/desktop_prepare.py", "scripts/bundles/native_prepared.py",
                     "apps/desktop/scripts/prepared-native-deps.mjs", "pm/build_operations.py"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        old = path.read_bytes() if path.exists() else b""
        path.write_bytes(old + b"\n")
        changed = describe_cache(source, cache, "desktop-production")
        assert changed["restore_keys"][0] != first["restore_keys"][0], relative
        assert changed["restore_keys"][1] == first["restore_keys"][1], "unaffected domains remain restorable"
        if old:
            path.write_bytes(old)
        else:
            path.unlink()

    monkeypatch.setenv("GITHUB_RUN_ID", "other-run")
    rerun = describe_cache(source, cache, "desktop-production")
    assert rerun["key"] != first["key"]
    assert rerun["restore_keys"] == first["restore_keys"]
    monkeypatch.setenv("ImageVersion", "different-image")
    assert describe_cache(source, cache, "desktop-production")["restore_keys"][1] != first["restore_keys"][1]
    assert describe_cache(source, cache, "payload-only")["restore_keys"][1] != rerun["restore_keys"][1]


def test_cache_roots_cannot_select_private_state_or_follow_workspace_symlinks(tmp_path, monkeypatch):
    from scripts.ci.desktop_build_cache import describe_cache

    source = source_tree(tmp_path / "source")
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    with pytest.raises(ValueError, match="HERMES_HOME"):
        describe_cache(source, home, "test")
    with pytest.raises(ValueError, match="path"):
        describe_cache(source, tmp_path / "*", "test")
    with pytest.raises(ValueError, match="producer"):
        describe_cache(source, tmp_path / "cache", "test\nOTHER=value")


def test_composite_transports_only_and_uses_the_restore_key_for_save(tmp_path):
    from ruamel.yaml import YAML

    action = ROOT / ".github/actions/desktop-build-cache/action.yml"
    assert action.is_file(), "the desktop transport action is missing"
    document = YAML(typ="safe").load(action.read_text(encoding="utf-8-sig"))
    steps = document["runs"]["steps"]
    source = source_tree(tmp_path / "source")
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    inputs = {"source": str(source), "cache": str(cache), "work": str(tmp_path / "job"),
              "producer": "desktop-production", "phase": "restore", "key": ""}

    describe_step = next(step for step in steps if step.get("id") == "describe")
    def run_description(phase, key):
        inputs.update(phase=phase, key=key)
        values = {f"${{{{ inputs.{name} }}}}": value for name, value in inputs.items()}
        values["${{ github.action_path }}"] = str(action.parent)
        env = {name: values[value] for name, value in describe_step["env"].items()}
        output.unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", "-e", "-c", describe_step["run"]], cwd=tmp_path,
            env={**os.environ, **env, "GITHUB_OUTPUT": str(output), "RUNNER_OS": "Linux"},
            capture_output=True, text=True, encoding="utf-8", timeout=30,
            check=False,
        )
        return result

    result = run_description("restore", "")
    assert result.returncode == 0, result.stdout + result.stderr
    values = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8-sig").splitlines())
    assert json.loads(values["cache-paths"])
    assert not cache.exists(), "description must not install/prune anything"
    restored_key = values["cache-key"]
    result = run_description("save", restored_key)
    assert result.returncode == 0, result.stdout + result.stderr
    assert run_description("save", "").returncode != 0
    assert run_description("typo", "").returncode != 0

    restore = next(step for step in steps if step.get("id") == "restore")
    save = next(step for step in steps if step.get("id") == "save")
    assert restore["uses"].startswith("actions/cache/restore@")
    assert save["uses"].startswith("actions/cache/save@")
    assert restore["with"]["path"] == save["with"]["path"]
    assert save["with"]["key"] == "${{ inputs.key }}"
    assert restore["continue-on-error"] is True
    assert save["continue-on-error"] is True
    for step in steps:
        assert not gate(step["if"], {}, {}, job_if=False, cancelled=True), \
            "failure salvage must not run during cancellation"
    # A cache service failure is reported instead of failing preparation.
    (report,) = [step for step in steps if "::warning::" in step.get("run", "")]
    for failed in ("restore", "save"):
        assert gate(report["if"], {}, {}, job_if=False, failed=True, steps={failed: {"outcome": "failure"}})
    assert not gate(report["if"], {}, {}, job_if=False, steps={"restore": {"outcome": "success"}})


def test_action_path_join_is_an_actual_newline(tmp_path):
    import re
    from ruamel.yaml import YAML

    action = YAML(typ="safe").load((ROOT / ".github/actions/desktop-build-cache/action.yml").read_text(encoding="utf-8-sig"))
    expression = next(step for step in action["runs"]["steps"] if step.get("id") == "restore")["with"]["path"]
    # GitHub expression strings do not process backslash escapes. A JSON
    # string does; evaluate that exact separator instead of trusting YAML.
    match = re.search(r", fromJSON\('([^']+)'\)\)", expression)
    assert match, "decode the JSON newline rather than joining paths with a literal backslash-n"
    assert json.loads(match[1]) == "\n"


def test_job_state_cannot_be_nested_in_a_transport_domain(tmp_path):
    from scripts.ci.desktop_build_cache import describe_cache

    source = source_tree(tmp_path / "source")
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="work"):
        describe_cache(source, cache, "test", work=cache / "native/job")


def test_private_home_and_source_cannot_be_children_of_cached_roots(tmp_path, monkeypatch):
    from scripts.ci.desktop_build_cache import describe_cache

    source = source_tree(tmp_path / "source")
    cache = tmp_path / "cache"
    monkeypatch.setenv("HERMES_HOME", str(cache / "native/private"))
    with pytest.raises(ValueError, match="HERMES_HOME"):
        describe_cache(source, cache, "test")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (cache / "tools").mkdir(parents=True)
    nested = source_tree(cache / "tools/source")
    with pytest.raises(ValueError, match="source"):
        describe_cache(nested, cache, "test")


def test_restored_npm_cache_installs_offline_and_excludes_job_secrets(tmp_path):
    """node_modules is not transported; npm's content cache alone must let a fresh
    checkout install offline, and job-local secrets never ride along."""
    import shutil
    from scripts.ci.desktop_build_cache import describe_cache

    node = shutil.which("node")
    assert node, "the transport integration test requires the build's Node prerequisite"
    registry_package = tmp_path / "left-pad"
    registry_package.mkdir()
    (registry_package / "package.json").write_text(json.dumps({"name": "left-pad", "version": "1.0.0"}), encoding="utf-8")
    owner = ROOT / "scripts/build/node-deps.mjs"

    def run(source, cache, *args, offline):
        env = {**os.environ, "npm_config_cache": str(cache / "npm"), "npm_config_offline": str(offline).lower()}
        return subprocess.run([node, str(owner), *args], cwd=source, env=env,
                              capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)

    source, cache = tmp_path / "old/source", tmp_path / "old/cache"
    (source / "editor").mkdir(parents=True)
    tarball = run(registry_package, cache, "--npm", "pack", "--pack-destination", str(tmp_path), offline=False)
    assert tarball.returncode == 0, tarball.stdout + tarball.stderr
    manifest = {"name": "cache-fixture", "private": True, "workspaces": ["editor"],
                "dependencies": {"left-pad": f"file:{tmp_path / 'left-pad-1.0.0.tgz'}"}}
    (source / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    (source / "editor/package.json").write_text(json.dumps({"name": "editor", "version": "1.0.0"}), encoding="utf-8")
    locked = run(source, cache, "--npm", "install", "--package-lock-only", "--no-audit", "--no-fund", offline=False)
    assert locked.returncode == 0, locked.stdout + locked.stderr
    prepared = run(source, cache, "--source", str(source), "--workspace", "editor", "--reuse", offline=False)
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    for path in (source / ".env", cache / "npm/_logs/secret.log", cache / "npm/.npmrc", cache / "prepared.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("planted-secret", encoding="utf-8")

    description = describe_cache(source, cache, "test")
    assert not any(Path(path).is_relative_to(source) for path in description["paths"])
    restored_source, restored_cache = tmp_path / "new/source", tmp_path / "new/cache"
    for name in ("package.json", "package-lock.json", "editor/package.json"):
        (restored_source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, restored_source / name)
    for name in description["paths"]:
        path = Path(name)
        if path.exists():
            shutil.copytree(path, restored_cache / path.relative_to(cache), symlinks=True)
    assert not any(b"planted-secret" in path.read_bytes() for path in (tmp_path / "new").rglob("*") if path.is_file())
    (tmp_path / "left-pad-1.0.0.tgz").unlink()  # only the transported content cache can satisfy it now
    warm = run(restored_source, restored_cache, "--source", str(restored_source), "--workspace", "editor", "--reuse", offline=True)
    assert warm.returncode == 0, warm.stdout + warm.stderr
    assert (restored_source / "node_modules/left-pad/package.json").is_file()
