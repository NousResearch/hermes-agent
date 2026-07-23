from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import termux_wheelhouse as tw


EXPECTED_NAMES = {
    "cffi",
    "cryptography",
    "jiter",
    "markupsafe",
    "pillow",
    "psutil",
    "pydantic-core",
    "pyyaml",
    "rpds-py",
    "ruamel-yaml-clib",
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_current_uv_lock_matches_immutable_release() -> None:
    root = Path(__file__).resolve().parents[2]
    tw.validate_uv_lock(root / "uv.lock")
    assert set(tw.LOCKED_WHEELS) == EXPECTED_NAMES
    assert tw.RELEASE_TAG == "wheelhouse-cp313-android24-arm64-20260719.1"
    assert tw.RELEASE_COMMIT == "b042ce9e662f14794ef6b5664c9ae711330df31b"


def test_uv_lock_pin_drift_is_rejected() -> None:
    text = """
[[package]]
name = "cffi"
version = "999.0"
"""
    with pytest.raises(tw.TermuxWheelhouseMismatch, match="cffi: expected 2.0.0"):
        tw.validate_uv_lock_text(text, source="remote uv.lock")


def test_resolved_requirements_pin_drift_is_rejected() -> None:
    requirements = "\n".join(
        f"{name}=={version}"
        for name, (version, _filename, _sha) in tw.LOCKED_WHEELS.items()
    )
    tw.validate_requirements_text(requirements)
    with pytest.raises(tw.TermuxWheelhouseMismatch, match="cryptography"):
        tw.validate_requirements_text(
            requirements.replace("cryptography==46.0.7", "cryptography==1")
        )


def test_runtime_requires_android_cp313_arm64_api24() -> None:
    tw.validate_runtime(
        python_version=(3, 13),
        machine="aarch64",
        android_api=35,
        sys_platform="android",
    )
    with pytest.raises(tw.TermuxWheelhouseUnsupported, match="CPython 3.13"):
        tw.validate_runtime(
            python_version=(3, 14),
            machine="aarch64",
            android_api=35,
            sys_platform="android",
        )
    with pytest.raises(tw.TermuxWheelhouseUnsupported, match="arm64"):
        tw.validate_runtime(
            python_version=(3, 13),
            machine="x86_64",
            android_api=35,
            sys_platform="android",
        )


def test_ensure_wheelhouse_verifies_stages_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {
        "alpha-1-cp313-cp313-android_24_arm64_v8a.whl": b"alpha-wheel",
        "beta-2-cp313-abi3-android_24_arm64_v8a.whl": b"beta-wheel",
    }
    locked = {
        "alpha": ("1", next(iter(files)), _digest(files[next(iter(files))])),
        "beta": ("2", list(files)[1], _digest(files[list(files)[1]])),
    }
    sums = "".join(
        f"{_digest(data)}  {name}\n" for name, data in files.items()
    ).encode()
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "SHA256SUMS").write_bytes(sums)
    for name, data in files.items():
        (remote / name).write_bytes(data)

    monkeypatch.setattr(tw, "LOCKED_WHEELS", locked)
    monkeypatch.setattr(tw, "RELEASE_TAG", "test-release")
    monkeypatch.setattr(tw, "RELEASE_COMMIT", "a" * 40)
    monkeypatch.setattr(tw, "RELEASE_BASE_URL", "https://example.invalid/test-release")
    monkeypatch.setattr(tw, "SHA256SUMS_SHA256", _digest(sums))
    downloads: list[str] = []

    def fake_download(_curl: str, url: str, destination: Path) -> None:
        downloads.append(url)
        shutil.copy2(remote / url.rsplit("/", 1)[-1], destination)

    monkeypatch.setattr(tw, "_download", fake_download)
    cache = tmp_path / "cache"
    first = tw.ensure_wheelhouse(cache, curl="curl", check_runtime=False)
    tw.verify_wheelhouse(first)
    assert len(downloads) == 3

    downloads.clear()
    second = tw.ensure_wheelhouse(cache, curl="curl", check_runtime=False)
    assert second == first
    assert downloads == []
    assert tw.binary_install_options(first) == [
        "--find-links",
        str(first),
        "--only-binary",
        ":all:",
    ]


def test_failed_refresh_preserves_previous_verified_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "alpha-1-cp313-cp313-android_24_arm64_v8a.whl"
    payload = b"known-good"
    sums = f"{_digest(payload)}  {filename}\n".encode()
    monkeypatch.setattr(
        tw, "LOCKED_WHEELS", {"alpha": ("1", filename, _digest(payload))}
    )
    monkeypatch.setattr(tw, "RELEASE_TAG", "test-release")
    monkeypatch.setattr(tw, "RELEASE_COMMIT", "b" * 40)
    monkeypatch.setattr(tw, "SHA256SUMS_SHA256", _digest(sums))

    cache = tmp_path / "cache" / "test-release"
    cache.mkdir(parents=True)
    (cache / "SHA256SUMS").write_bytes(sums)
    (cache / filename).write_bytes(payload)
    (cache / "release.json").write_text(
        json.dumps(tw._release_marker(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tw.verify_wheelhouse(cache)

    # Corrupting the cached wheel forces a refresh; a failed staged download
    # must not rename or delete the existing directory.
    (cache / filename).write_bytes(b"corrupt-cache")

    def fail_download(_curl: str, _url: str, _destination: Path) -> None:
        raise OSError("network failed")

    monkeypatch.setattr(tw, "_download", fail_download)
    with pytest.raises(OSError, match="network failed"):
        tw.ensure_wheelhouse(tmp_path / "cache", curl="curl", check_runtime=False)
    assert cache.is_dir()
    assert (cache / filename).read_bytes() == b"corrupt-cache"
    assert not list((tmp_path / "cache").glob(".test-release.tmp-*"))


def test_post_pull_gate_rolls_back_on_verifier_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli import main as hm

    project = tmp_path / "repo"
    (project / "scripts").mkdir(parents=True)
    (project / "venv" / "bin").mkdir(parents=True)
    (project / "venv" / "bin" / "python").write_text("", encoding="utf-8")
    (project / "scripts" / "prepare_termux_wheelhouse.py").write_text(
        "", encoding="utf-8"
    )
    (project / "uv.lock").write_text("", encoding="utf-8")
    monkeypatch.setattr(hm, "PROJECT_ROOT", project)
    monkeypatch.setattr(hm, "_is_termux_env", lambda env=None: True)
    monkeypatch.setattr(hm, "_is_android_python", lambda: True)
    monkeypatch.setattr(
        hm, "_termux_wheelhouse_cache_root", lambda env=None: tmp_path / "cache"
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs):
        calls.append(command)
        if "prepare_termux_wheelhouse.py" in " ".join(command):
            return SimpleNamespace(returncode=1, stdout="", stderr="pin mismatch\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hm.subprocess, "run", fake_run)
    with pytest.raises(SystemExit):
        hm._termux_post_pull_wheelhouse_gate(["git"], "1" * 40)
    assert ["git", "reset", "--hard", "1" * 40] in calls


def test_post_pull_gate_requires_binary_only_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli import main as hm

    project = tmp_path / "repo"
    scripts = project / "scripts"
    venv_bin = project / "venv" / "bin"
    scripts.mkdir(parents=True)
    venv_bin.mkdir(parents=True)
    target_python = venv_bin / "python"
    target_python.write_text("", encoding="utf-8")
    for relative in (
        "scripts/prepare_termux_wheelhouse.py",
        "scripts/termux_requirements.py",
        "pyproject.toml",
        "uv.lock",
        "constraints-termux.txt",
    ):
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("", encoding="utf-8")

    monkeypatch.setattr(hm, "PROJECT_ROOT", project)
    monkeypatch.setattr(hm, "_is_termux_env", lambda env=None: True)
    monkeypatch.setattr(hm, "_is_android_python", lambda: True)
    monkeypatch.setattr(hm.shutil, "which", lambda name: "/termux/bin/uv")
    monkeypatch.setattr(
        hm, "_termux_wheelhouse_cache_root", lambda env=None: tmp_path / "cache"
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs):
        calls.append(command)
        joined = " ".join(command)
        if "prepare_termux_wheelhouse.py" in joined:
            return SimpleNamespace(
                returncode=0,
                stdout=str(tmp_path / "cache" / tw.RELEASE_TAG) + "\n",
                stderr="",
            )
        if command[:2] == [str(target_python), "-c"]:
            return SimpleNamespace(returncode=0, stdout="3.13.14\n", stderr="")
        if "termux_requirements.py" in joined:
            Path(command[command.index("--requirements") + 1]).write_text(
                "cffi==2.0.0\n", encoding="utf-8"
            )
            Path(command[command.index("--constraints") + 1]).write_text(
                "", encoding="utf-8"
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "compile" in command:
            Path(command[command.index("--output-file") + 1]).write_text(
                "cffi==2.0.0\n", encoding="utf-8"
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "--dry-run" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hm.subprocess, "run", fake_run)
    hm._termux_post_pull_wheelhouse_gate(["git"], "1" * 40)

    dry_runs = [command for command in calls if "--dry-run" in command]
    assert len(dry_runs) == 1
    assert "--reinstall" in dry_runs[0]
    assert "--only-binary" in dry_runs[0]
    assert ":all:" in dry_runs[0]
    assert "--find-links" in dry_runs[0]
    assert not any(command[:3] == ["git", "reset", "--hard"] for command in calls)


def test_termux_group_uses_wheelhouse_without_psutil_source_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import main as hm

    calls: list[tuple[list[str], dict[str, str] | None]] = []
    monkeypatch.setattr(hm, "_is_termux_env", lambda env=None: True)
    monkeypatch.setattr(hm, "_is_android_python", lambda: True)
    monkeypatch.setattr(
        hm,
        "_install_termux_dependencies_from_wheelhouse",
        lambda prefix, env=None: calls.append((prefix, env)) or True,
    )
    monkeypatch.setattr(
        hm,
        "_install_psutil_android_compat",
        lambda *_args, **_kwargs: pytest.fail("psutil source fallback must not run"),
    )
    monkeypatch.setattr(
        hm,
        "_run_quarantined_install",
        lambda *_args, **_kwargs: pytest.fail("generic editable resolver must not run"),
    )

    env = {"PREFIX": "/data/data/com.termux/files/usr"}
    hm._install_python_dependencies_with_optional_fallback(
        ["/data/data/com.termux/files/usr/bin/uv", "pip"],
        env=env,
        group="termux-all",
    )

    assert calls == [(["/data/data/com.termux/files/usr/bin/uv", "pip"], env)]
