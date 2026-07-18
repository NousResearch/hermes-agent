from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from scripts.canary import full_canary_owner_launcher as launcher


RELEASE_SHA = "a" * 40
LAUNCHER_SHA256 = "b" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sdk_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, tuple[int, int, str]]:
    monkeypatch.setattr(launcher, "_canonical_owner_home", lambda: str(tmp_path))
    monkeypatch.setattr(
        launcher,
        "_current_launcher_sha256",
        lambda: LAUNCHER_SHA256,
    )
    hermes = tmp_path / ".hermes"
    trusted = hermes / "trusted"
    hermes.mkdir(mode=0o700)
    trusted.mkdir(mode=0o700)
    sdk = tmp_path / launcher._TRUSTED_SDK_RELATIVE
    (sdk / "bin").mkdir(parents=True, mode=0o755)
    (sdk / "lib/a").mkdir(parents=True, mode=0o755)
    (sdk / "lib/b").mkdir(mode=0o755)
    (sdk / "lib/c").mkdir(mode=0o755)
    (sdk / "VERSION").write_bytes(
        f"{launcher._GCLOUD_SDK_VERSION}\n".encode("ascii")
    )
    (sdk / "bin/gcloud").write_bytes(b"#!/bin/sh\nexit 0\n")
    (sdk / "bin/gcloud").chmod(0o755)
    (sdk / "lib/gcloud.py").write_bytes(b"# pinned gcloud module\n")
    for name in ("a", "b", "c"):
        (sdk / f"lib/{name}/module.py").write_bytes(
            f"NAME = {name!r}\n".encode("ascii")
        )
    publication_tree = launcher._capture_sdk_publication_tree(str(sdk))
    unsigned = {
        "schema": launcher.TRUSTED_SDK_PUBLICATION_INTENT_SCHEMA,
        "ok": True,
        "state": "trusted_sdk_publication_prepared",
        "publication_release_sha": RELEASE_SHA,
        "launcher_sha256": "0" * 64,
        "sdk_archive_url": launcher._GCLOUD_SDK_ARCHIVE_URL,
        "sdk_archive_bytes": launcher._GCLOUD_SDK_ARCHIVE_BYTES,
        "sdk_archive_sha256": launcher._GCLOUD_SDK_ARCHIVE_SHA256,
        "sdk_version": launcher._GCLOUD_SDK_VERSION,
        "sdk_root": str(sdk),
        "sdk_tree_entries": publication_tree[0],
        "sdk_tree_bytes": publication_tree[1],
        "sdk_tree_sha256": publication_tree[2],
        "prepared_at_unix": 100,
    }
    intent = {
        **unsigned,
        "intent_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }
    intent_path = tmp_path / launcher._TRUSTED_SDK_PUBLICATION_INTENT_RELATIVE
    intent_path.write_bytes(_canonical(intent) + b"\n")
    intent_path.chmod(0o600)
    return sdk, intent_path, publication_tree


def _add_cache_file(sdk: Path, package: str, name: str, payload: bytes) -> Path:
    cache = sdk / f"lib/{package}/__pycache__"
    cache.mkdir(exist_ok=True)
    path = cache / name
    path.write_bytes(payload)
    return path


def test_repair_removes_exact_630_bytecode_files_and_replays_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, _intent_path, publication_tree = _sdk_fixture(tmp_path, monkeypatch)
    packages = ("a", "b", "c")
    expected_bytes = 0
    for index in range(630):
        payload = f"cpython-313-bytecode-{index}".encode("ascii")
        expected_bytes += len(payload)
        _add_cache_file(
            sdk,
            packages[index % len(packages)],
            f"module_{index}.cpython-313.pyc",
            payload,
        )
    real_fsync = launcher.os.fsync
    fsynced: list[int] = []

    def recording_fsync(descriptor: int) -> None:
        fsynced.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(launcher.os, "fsync", recording_fsync)

    receipt = launcher.repair_trusted_gcloud_sdk_bytecode(
        RELEASE_SHA,
        now_unix=1_000,
        launcher_sha256=LAUNCHER_SHA256,
    )

    assert receipt["schema"] == launcher.TRUSTED_SDK_BYTECODE_REPAIR_RECEIPT_SCHEMA
    assert receipt["bytecode_files_removed"] == 630
    assert receipt["bytecode_bytes_removed"] == expected_bytes
    assert receipt["cache_directories_removed"] == 3
    assert receipt["sdk_publication_tree_sha256"] == publication_tree[2]
    assert not tuple(sdk.rglob("__pycache__"))
    assert not tuple(sdk.rglob("*.pyc"))
    assert launcher._capture_sdk_publication_tree(str(sdk)) == publication_tree
    assert len(fsynced) >= 11

    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    assert receipt["receipt_sha256"] == hashlib.sha256(
        _canonical(unsigned)
    ).hexdigest()
    receipt_path = Path(
        launcher._trusted_sdk_bytecode_repair_receipt_path(
            str(tmp_path),
            RELEASE_SHA,
        )
    )
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert receipt_path.read_bytes() == _canonical(receipt) + b"\n"

    fsync_count = len(fsynced)
    replay = launcher.repair_trusted_gcloud_sdk_bytecode(
        RELEASE_SHA,
        now_unix=9_999,
        launcher_sha256=LAUNCHER_SHA256,
    )
    assert replay == receipt
    assert len(fsynced) == fsync_count


def test_repair_recovers_from_partial_unlink_with_original_receipt_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, _intent_path, _publication_tree = _sdk_fixture(tmp_path, monkeypatch)
    for index in range(3):
        _add_cache_file(
            sdk,
            "a",
            f"module_{index}.cpython-313.pyc",
            f"bytecode-{index}".encode("ascii"),
        )
    real_unlink = launcher.os.unlink
    calls = 0

    def interrupted_unlink(path: str, *, dir_fd: int | None = None) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected interruption")
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(launcher.os, "unlink", interrupted_unlink)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_sdk_bytecode_repair_remove_failed",
    ):
        launcher.repair_trusted_gcloud_sdk_bytecode(
            RELEASE_SHA,
            now_unix=1_000,
            launcher_sha256=LAUNCHER_SHA256,
        )
    repair_intent_path = Path(
        launcher._trusted_sdk_bytecode_repair_intent_path(
            str(tmp_path),
            RELEASE_SHA,
        )
    )
    assert repair_intent_path.is_file()
    assert len(tuple(sdk.rglob("*.pyc"))) == 2

    monkeypatch.setattr(launcher.os, "unlink", real_unlink)
    receipt = launcher.repair_trusted_gcloud_sdk_bytecode(
        RELEASE_SHA,
        now_unix=2_000,
        launcher_sha256=LAUNCHER_SHA256,
    )
    assert receipt["bytecode_files_removed"] == 3
    assert receipt["cache_directories_removed"] == 1
    assert not tuple(sdk.rglob("*.pyc"))


def test_repair_recovers_when_receipt_write_failed_after_all_deletions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, _intent_path, _publication_tree = _sdk_fixture(tmp_path, monkeypatch)
    _add_cache_file(sdk, "a", "module.cpython-313.pyc", b"bytecode")
    real_write = launcher._write_owner_file_no_replace

    def fail_receipt(destination: str, payload: bytes, **kwargs: object) -> None:
        if "trusted-sdk-bytecode-repair-569.0.0-" in destination:
            raise launcher.OwnerLauncherError(
                "trusted_sdk_bytecode_repair_receipt_write_failed"
            )
        real_write(destination, payload, **kwargs)

    monkeypatch.setattr(launcher, "_write_owner_file_no_replace", fail_receipt)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_sdk_bytecode_repair_receipt_write_failed",
    ):
        launcher.repair_trusted_gcloud_sdk_bytecode(
            RELEASE_SHA,
            now_unix=1_000,
            launcher_sha256=LAUNCHER_SHA256,
        )
    assert not tuple(sdk.rglob("__pycache__"))
    assert Path(
        launcher._trusted_sdk_bytecode_repair_intent_path(
            str(tmp_path),
            RELEASE_SHA,
        )
    ).is_file()

    monkeypatch.setattr(launcher, "_write_owner_file_no_replace", real_write)
    receipt = launcher.repair_trusted_gcloud_sdk_bytecode(
        RELEASE_SHA,
        now_unix=2_000,
        launcher_sha256=LAUNCHER_SHA256,
    )
    assert receipt["bytecode_files_removed"] == 1
    assert receipt["cache_directories_removed"] == 1


def test_descriptor_relative_delete_rejects_name_swap_before_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, _intent_path, _publication_tree = _sdk_fixture(tmp_path, monkeypatch)
    original = _add_cache_file(
        sdk,
        "a",
        "module.cpython-313.pyc",
        b"original-bytecode",
    )
    replacement = original.with_name("replacement.pyc")
    replacement.write_bytes(b"replacement-bytecode")
    real_stat = launcher.os.stat
    swapped = False

    def swapping_stat(
        path: str | int,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal swapped
        if path == original.name and dir_fd is not None and not swapped:
            swapped = True
            original.rename(original.with_name("audited-moved.pyc"))
            replacement.rename(original)
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(launcher.os, "stat", swapping_stat)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_sdk_bytecode_repair_tree_changed",
    ):
        launcher.repair_trusted_gcloud_sdk_bytecode(
            RELEASE_SHA,
            now_unix=1_000,
            launcher_sha256=LAUNCHER_SHA256,
        )
    assert original.read_bytes() == b"replacement-bytecode"


@pytest.mark.parametrize(
    "artifact",
    ("other_file", "symlink", "nested_directory", "optimized_bytecode"),
)
def test_repair_rejects_other_cache_content_before_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    sdk, _intent_path, _publication_tree = _sdk_fixture(tmp_path, monkeypatch)
    valid = _add_cache_file(sdk, "a", "valid.cpython-313.pyc", b"valid")
    cache = valid.parent
    if artifact == "other_file":
        (cache / "README.txt").write_text("not bytecode", encoding="ascii")
    elif artifact == "symlink":
        (cache / "linked.cpython-313.pyc").symlink_to(sdk / "lib/gcloud.py")
    elif artifact == "nested_directory":
        (cache / "nested").mkdir()
    else:
        (cache / "module.pyo").write_bytes(b"optimized")

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_sdk_bytecode_repair_scope_invalid",
    ):
        launcher.repair_trusted_gcloud_sdk_bytecode(
            RELEASE_SHA,
            now_unix=1_000,
            launcher_sha256=LAUNCHER_SHA256,
        )

    assert valid.read_bytes() == b"valid"
    assert cache.is_dir()


def test_repair_rejects_hardlinked_or_group_writable_bytecode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, _intent_path, _publication_tree = _sdk_fixture(tmp_path, monkeypatch)
    source = sdk / "lib/a/module.py"
    cache = sdk / "lib/a/__pycache__"
    cache.mkdir()
    linked = cache / "module.cpython-313.pyc"
    os.link(source, linked)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_sdk_bytecode_repair_scope_invalid",
    ):
        launcher.repair_trusted_gcloud_sdk_bytecode(
            RELEASE_SHA,
            now_unix=1_000,
            launcher_sha256=LAUNCHER_SHA256,
        )
    assert linked.exists()

    linked.unlink()
    unsafe = cache / "module.cpython-313.pyc"
    unsafe.write_bytes(b"unsafe")
    unsafe.chmod(0o666)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_sdk_bytecode_repair_scope_invalid",
    ):
        launcher.repair_trusted_gcloud_sdk_bytecode(
            RELEASE_SHA,
            now_unix=1_000,
            launcher_sha256=LAUNCHER_SHA256,
        )
    assert unsafe.exists()


def test_repair_rejects_bytecode_outside_cache_and_publication_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, _intent_path, _publication_tree = _sdk_fixture(tmp_path, monkeypatch)
    cache_file = _add_cache_file(
        sdk,
        "a",
        "module.cpython-313.pyc",
        b"valid-cache-bytecode",
    )
    outside = sdk / "lib/outside.pyc"
    outside.write_bytes(b"outside")

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_sdk_bytecode_repair_scope_invalid",
    ):
        launcher.repair_trusted_gcloud_sdk_bytecode(
            RELEASE_SHA,
            now_unix=1_000,
            launcher_sha256=LAUNCHER_SHA256,
        )
    assert cache_file.exists()

    outside.unlink()
    (sdk / "lib/gcloud.py").write_bytes(b"drifted publication\n")
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_runtime_publication_intent_invalid",
    ):
        launcher.repair_trusted_gcloud_sdk_bytecode(
            RELEASE_SHA,
            now_unix=1_000,
            launcher_sha256=LAUNCHER_SHA256,
        )
    assert cache_file.exists()


def test_repair_accepts_only_exact_sdk_and_private_owner_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, _intent_path, _publication_tree = _sdk_fixture(tmp_path, monkeypatch)
    cache_file = _add_cache_file(
        sdk,
        "a",
        "module.cpython-313.pyc",
        b"valid-cache-bytecode",
    )
    (tmp_path / ".hermes/trusted").chmod(0o755)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_runtime_directory_invalid",
    ):
        launcher.repair_trusted_gcloud_sdk_bytecode(
            RELEASE_SHA,
            now_unix=1_000,
            launcher_sha256=LAUNCHER_SHA256,
        )
    assert cache_file.exists()


def test_repair_replay_rejects_tampered_or_linked_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, _intent_path, _publication_tree = _sdk_fixture(tmp_path, monkeypatch)
    _add_cache_file(sdk, "a", "module.cpython-313.pyc", b"bytecode")
    receipt = launcher.repair_trusted_gcloud_sdk_bytecode(
        RELEASE_SHA,
        now_unix=1_000,
        launcher_sha256=LAUNCHER_SHA256,
    )
    receipt_path = Path(
        launcher._trusted_sdk_bytecode_repair_receipt_path(
            str(tmp_path),
            RELEASE_SHA,
        )
    )
    source = receipt_path.with_name("repair-receipt-source.json")
    receipt_path.rename(source)
    os.link(source, receipt_path)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_sdk_bytecode_repair_receipt_invalid",
    ):
        launcher.repair_trusted_gcloud_sdk_bytecode(
            RELEASE_SHA,
            now_unix=2_000,
            launcher_sha256=LAUNCHER_SHA256,
        )
    assert json.loads(source.read_text(encoding="utf-8")) == receipt


def test_repair_cli_is_explicit_pre_runtime_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    receipt = {"ok": True, "receipt_sha256": "c" * 64}
    monkeypatch.setattr(
        launcher,
        "require_trusted_bootstrap_interpreter",
        lambda: calls.append("bootstrap_interpreter"),
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda release: calls.append(("provenance", release)) or LAUNCHER_SHA256,
    )
    monkeypatch.setattr(
        launcher,
        "repair_trusted_gcloud_sdk_bytecode",
        lambda release, *, launcher_sha256: calls.append(
            ("repair", release, launcher_sha256)
        )
        or receipt,
    )
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_runtime",
        lambda _release: pytest.fail("repair reached normal runtime"),
    )
    monkeypatch.setattr(
        launcher,
        "_emit_canonical_line",
        lambda value: calls.append(("emit", value)),
    )

    result = launcher.main([
        "--release-sha",
        RELEASE_SHA,
        "--repair-trusted-sdk-bytecode",
    ])

    assert result == 0
    assert calls.count("bootstrap_interpreter") == 2
    assert calls.count(("provenance", RELEASE_SHA)) == 2
    assert ("repair", RELEASE_SHA, LAUNCHER_SHA256) in calls
    assert ("emit", receipt) in calls
    with pytest.raises(SystemExit):
        launcher._cli_parser().parse_args([
            "--release-sha",
            RELEASE_SHA,
            "--repair-trusted-sdk-bytecode",
            "--bootstrap-trusted-runtime",
        ])


def test_repair_cli_rejects_external_iam_before_any_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(
        launcher,
        "require_trusted_bootstrap_interpreter",
        lambda: pytest.fail("invalid repair CLI reached runtime"),
    )
    monkeypatch.setattr(launcher, "_emit_canonical_line", emitted.append)

    result = launcher.main([
        "--release-sha",
        RELEASE_SHA,
        "--repair-trusted-sdk-bytecode",
        "--external-iam-policy-sha256",
        "d" * 64,
    ])

    assert result == 2
    assert emitted[0]["error_code"] == "trusted_sdk_bytecode_repair_cli_invalid"
