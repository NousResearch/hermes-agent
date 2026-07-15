from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from gateway import production_runtime_dependencies as package
from scripts.canary import package_production_runtime_dependencies as legacy_package


ROOT = Path(__file__).parents[3]
REVISION = "a" * 40


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _minimal_release(tmp_path: Path) -> Path:
    release = (tmp_path / f"hermes-agent-{REVISION[:12]}").resolve()
    release.mkdir()
    (release / ".codex-source-commit").write_text(
        REVISION + "\n",
        encoding="ascii",
    )
    return release


def test_legacy_source_wrapper_resolves_to_the_packaged_implementation() -> None:
    assert legacy_package is package
    wrapper = ROOT / "scripts/canary/package_production_runtime_dependencies.py"
    source = wrapper.read_text(encoding="utf-8")
    assert "gateway import production_runtime_dependencies" in source
    assert "urllib.request" not in source
    assert "def install_release_dependencies" not in source
    completed = subprocess.run(
        (str(wrapper), "--help"),
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "{prepare,install,build-manifest,verify}" in completed.stdout


def test_exact_release_local_node_browser_and_ddgs_contract() -> None:
    assert package.AGENT_BROWSER_VERSION == "0.26.0"
    assert package.NODE_VERSION == "24.18.0"
    assert package.NODE_ARCHIVE_SHA256 == (
        "55aa7153f9d88f28d765fcdad5ae6945b5c0f98a36881703817e4c450fa76742"
    )
    assert package.CHROME_VERSION == "150.0.7871.114"
    assert package.CHROME_ARCHIVE_SHA256 == (
        "03963c0dd9bf91e9b0e760cff37680f9b92ff42758182286382787622323cf9d"
    )
    assert package.CHROME_EXECUTABLE == Path(
        "ops/muncho/runtime/dependencies/chrome-linux64/chrome"
    )
    assert package.NODE_EXECUTABLE == Path(
        "ops/muncho/runtime/dependencies/node-linux-x64/bin/node"
    )
    assert package.AGENT_BROWSER_CONFIG == Path(
        "ops/muncho/runtime/dependencies/agent-browser.json"
    )
    assert package.AGENT_BROWSER_CONFIG_BYTES == b"{}\n"
    assert package.DDGS_LOCKED_DISTRIBUTIONS["ddgs"] == "9.14.4"
    assert len(package.DDGS_LOCKED_DISTRIBUTIONS) == 17


def test_committed_locks_close_every_ddgs_wheel_and_agent_browser() -> None:
    requirements, contract = package._locked_python_requirements(ROOT)
    node = package._validate_node_lock(ROOT)

    assert set(contract["distributions"]) == set(
        package.DDGS_LOCKED_DISTRIBUTIONS
    )
    assert requirements.count("\n") == len(package.DDGS_LOCKED_DISTRIBUTIONS)
    assert all("--hash=sha256:" in line for line in requirements.splitlines())
    assert node["version"] == package.AGENT_BROWSER_VERSION
    assert node["integrity"] == package.AGENT_BROWSER_INTEGRITY


def test_ddgs_extra_drift_is_rejected(tmp_path: Path) -> None:
    release = tmp_path / "source"
    release.mkdir()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    (release / "pyproject.toml").write_text(
        pyproject.replace("ddgs==9.14.4", "ddgs==9.14.3", 1),
        encoding="utf-8",
    )
    (release / "uv.lock").write_bytes((ROOT / "uv.lock").read_bytes())

    with pytest.raises(
        package.RuntimeDependencyError,
        match="runtime_dependency_python_pin_invalid",
    ):
        package._locked_python_requirements(release)


def test_agent_browser_lock_drift_is_rejected(tmp_path: Path) -> None:
    release = tmp_path / "source"
    release.mkdir()
    (release / "package.json").write_bytes((ROOT / "package.json").read_bytes())
    lock = (ROOT / "package-lock.json").read_text(encoding="utf-8")
    (release / "package-lock.json").write_text(
        lock.replace(package.AGENT_BROWSER_INTEGRITY, "sha512-invalid", 1),
        encoding="utf-8",
    )

    with pytest.raises(
        package.RuntimeDependencyError,
        match="runtime_dependency_agent_browser_pin_invalid",
    ):
        package._validate_node_lock(release)


def test_python_install_disables_cache_and_removes_hash_requirements_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    interpreter = release / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"python")
    observed: list[dict[str, object]] = []

    def run(arguments, *, cwd, timeout, extra_environment=None):
        observed.append({
            "arguments": tuple(arguments),
            "cwd": cwd,
            "timeout": timeout,
            "environment": extra_environment,
        })
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(package, "_release_interpreter", lambda _root: interpreter)
    monkeypatch.setattr(package, "_run", run)
    package._install_python(release, "ddgs==9.14.4 --hash=sha256:" + "a" * 64 + "\n")

    assert observed[0] == {
        "arguments": (
            str(interpreter),
            "-I",
            "-m",
            "ensurepip",
            "--upgrade",
            "--default-pip",
        ),
        "cwd": release,
        "timeout": 120,
        "environment": None,
    }
    arguments = observed[1]["arguments"]
    assert "--no-cache-dir" in arguments
    requirement = Path(arguments[arguments.index("--requirement") + 1])
    assert not requirement.exists()
    assert not (release / ".cache").exists()


def test_npm_cache_is_exact_bounded_and_disposable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    observed: dict[str, object] = {}

    def run(arguments, *, cwd, timeout, extra_environment=None):
        observed.update({
            "arguments": tuple(arguments),
            "cwd": cwd,
            "timeout": timeout,
            "environment": dict(extra_environment or {}),
        })
        cache = Path(extra_environment["npm_config_cache"])
        (cache / "_cacache").mkdir()
        (cache / "_cacache/content").write_bytes(b"bounded")
        native = release / package.AGENT_BROWSER_NATIVE
        wrapper = release / package.AGENT_BROWSER_WRAPPER
        native.parent.mkdir(parents=True)
        native.write_bytes(b"native")
        wrapper.write_bytes(b"wrapper")
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(package, "_run", run)
    package._install_node(release)

    cache = release / package.NPM_CACHE_RELATIVE_PATH
    assert observed["cwd"] == release
    assert observed["timeout"] == 600
    assert observed["environment"] == {
        "npm_config_cache": str(cache),
        "npm_config_update_notifier": "false",
    }
    assert not cache.exists()
    assert not (release / ".npm").exists()


def test_oversized_npm_cache_fails_closed_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    monkeypatch.setattr(package, "NPM_CACHE_MAX_BYTES", 1)

    def run(arguments, *, cwd, timeout, extra_environment=None):
        del cwd, timeout
        cache = Path(extra_environment["npm_config_cache"])
        (cache / "oversized").write_bytes(b"xx")
        native = release / package.AGENT_BROWSER_NATIVE
        wrapper = release / package.AGENT_BROWSER_WRAPPER
        native.parent.mkdir(parents=True)
        native.write_bytes(b"native")
        wrapper.write_bytes(b"wrapper")
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(package, "_run", run)
    with pytest.raises(
        package.RuntimeDependencyError,
        match="runtime_dependency_npm_cache_oversized",
    ):
        package._install_node(release)

    assert (release / package.NPM_CACHE_RELATIVE_PATH).is_dir()


def test_agent_browser_config_is_exact_atomic_and_read_only(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    path = release / package.AGENT_BROWSER_CONFIG
    path.parent.mkdir(parents=True)
    path.write_text('{"proxy":"http://127.0.0.1:8080"}\n', encoding="ascii")

    package._install_agent_browser_config(release)

    assert path.read_bytes() == b"{}\n"
    assert path.stat().st_mode & 0o777 == 0o444
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_owner_preparation_cannot_claim_root_sealed_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release = _minimal_release(tmp_path)
    final = release
    for name in (
        "_validate_supported_platform",
        "_install_python",
        "_install_node_runtime",
        "_install_node",
        "_install_chrome",
    ):
        monkeypatch.setattr(package, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        package,
        "_locked_python_requirements",
        lambda _release: ("", {}),
    )
    monkeypatch.setattr(package, "_validate_node_lock", lambda _release: {})

    receipt = package.prepare_release_dependencies(
        release,
        REVISION,
        release_address=final,
    )

    path = release / package.AGENT_BROWSER_CONFIG
    assert receipt["schema"] == package.PREPARATION_SCHEMA
    assert receipt["root_seal_required"] is True
    assert receipt["agent_browser_config"]["owner_uid"] == os.geteuid()
    assert path.read_bytes() == package.AGENT_BROWSER_CONFIG_BYTES
    with pytest.raises(
        package.RuntimeDependencyError,
        match="runtime_dependency_agent_browser_config_invalid",
    ):
        package._agent_browser_config_identity(
            release,
            expected_uid=os.geteuid() + 1,
            expected_gid=os.getegid(),
        )


def test_root_sealed_config_requires_exact_owner_group_mode_and_one_link(
    tmp_path: Path,
) -> None:
    release = _minimal_release(tmp_path)
    package._install_agent_browser_config(release)
    path = release / package.AGENT_BROWSER_CONFIG

    observed = package._agent_browser_config_identity(
        release,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )
    assert observed["regular_one_link"] is True

    sibling = path.with_name("agent-browser-hardlink.json")
    os.link(path, sibling)
    with pytest.raises(
        package.RuntimeDependencyError,
        match="runtime_dependency_agent_browser_config_invalid",
    ):
        package._agent_browser_config_identity(
            release,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
        )


def test_verify_manifest_is_observational_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release = _minimal_release(tmp_path)
    unsigned = {
        "schema": package.MANIFEST_SCHEMA,
        "release_revision": REVISION,
        "secret_material_recorded": False,
    }
    expected = {
        **unsigned,
        "manifest_sha256": package._sha256(_canonical(unsigned)),
    }
    path = release / package.MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    original = _canonical(expected) + b"\n"
    path.write_bytes(original)
    observed = {**expected, "secret_material_recorded": True}
    monkeypatch.setattr(
        package,
        "_manifest_value",
        lambda *_args, **_kwargs: observed,
    )

    with pytest.raises(
        package.RuntimeDependencyError,
        match="runtime_dependency_manifest_drifted",
    ):
        package.verify_manifest(release, REVISION)

    assert path.read_bytes() == original


def test_same_parent_production_staging_is_final_address_bound(
    tmp_path: Path,
) -> None:
    final = (tmp_path / f"hermes-agent-{REVISION[:12]}").resolve()
    staging = (
        tmp_path / f".hermes-agent-{REVISION[:12]}.tmp.12345"
    ).resolve()
    staging.mkdir()
    (staging / ".codex-source-commit").write_text(
        REVISION + "\n",
        encoding="ascii",
    )

    observed, address = package._release_location(
        staging,
        REVISION,
        final,
    )

    assert observed == staging
    assert address == final
    escaped = (tmp_path / "other" / staging.name).resolve()
    escaped.mkdir(parents=True)
    (escaped / ".codex-source-commit").write_text(
        REVISION + "\n",
        encoding="ascii",
    )
    with pytest.raises(
        package.RuntimeDependencyError,
        match="runtime_dependency_release_address_invalid",
    ):
        package._release_location(escaped, REVISION, final)


def test_exact_canary_cli_uses_its_single_release_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = (tmp_path / "muncho-canary-releases").resolve()
    release = base / REVISION
    interpreter = release / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"python")
    (release / ".codex-source-commit").write_text(
        REVISION + "\n",
        encoding="ascii",
    )
    monkeypatch.setattr(package, "CANARY_RELEASE_BASE", base)
    monkeypatch.setattr(package.sys, "executable", str(interpreter))
    captured: dict[str, object] = {}

    def install(root: Path, revision: str, *, release_address=None):
        captured.update(
            {
                "root": root,
                "revision": revision,
                "address": release_address,
                "interpreter": package._release_interpreter(root),
            }
        )
        package._release_location(root, revision, release_address)
        return {
            "schema": package.MANIFEST_SCHEMA,
            "release_revision": revision,
            "manifest_sha256": "f" * 64,
        }

    monkeypatch.setattr(package, "install_release_dependencies", install)

    assert package._main(
        [
            "install",
            "--release-root",
            str(release),
            "--revision",
            REVISION,
        ]
    ) == 0
    assert captured == {
        "root": release,
        "revision": REVISION,
        "address": None,
        "interpreter": interpreter,
    }
    assert capsys.readouterr().out.endswith("\n")
