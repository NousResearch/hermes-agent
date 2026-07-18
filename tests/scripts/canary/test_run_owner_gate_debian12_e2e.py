from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.canary import run_owner_gate_debian12_e2e as harness


def _manifest() -> dict[str, object]:
    return {
        "bootstrap_pip": {"filename": "pip.whl"},
        "wheels": [{"filename": "runtime.whl"}],
    }


def _fake_docker_cp(
    arguments: tuple[str, ...],
    *,
    input_text: str | None = None,
    timeout: int = 300,
) -> str:
    del input_text, timeout
    assert arguments[:2] == ("docker", "cp")
    destination = Path(arguments[-1])
    if arguments[-2].endswith("owner-gate-wheelhouse/."):
        (destination / "pip.whl").write_bytes(b"pip")
        (destination / "runtime.whl").write_bytes(b"runtime")
    else:
        destination.write_bytes(
            harness.package.foundation.canonical_json_bytes(_manifest()) + b"\n"
        )
    return ""


def test_verified_wheelhouse_is_published_only_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    observed: dict[str, object] = {}

    def validate(*, root: Path, manifest: object) -> tuple[object, ...]:
        observed.update(root=root, manifest=manifest)
        return ()

    monkeypatch.setattr(harness, "_run", _fake_docker_cp)
    monkeypatch.setattr(harness.package, "validate_wheelhouse", validate)

    harness._publish_verified_wheelhouse(
        container="fixed-container",
        destination=destination,
    )

    assert observed["manifest"] == _manifest()
    observed_root = observed["root"]
    assert isinstance(observed_root, Path)
    assert observed_root.name == "artifacts"
    assert observed_root.parent.name.startswith(".published.stage-")
    assert {item.name for item in (destination / "artifacts").iterdir()} == {
        "pip.whl",
        "runtime.whl",
    }
    assert (destination / "wheelhouse-manifest.json").is_file()
    assert os.stat(destination).st_mode & 0o777 == 0o555
    assert os.stat(destination / "artifacts").st_mode & 0o777 == 0o555
    assert all(
        os.stat(item).st_mode & 0o777 == 0o444
        for item in (destination / "artifacts").iterdir()
    )


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    marker = destination / "keep"
    marker.write_text("untouched", encoding="ascii")

    with pytest.raises(
        harness.HarnessError,
        match="^owner_gate_debian12_wheelhouse_output_invalid$",
    ):
        harness._publish_verified_wheelhouse(
            container="fixed-container",
            destination=destination,
        )

    assert marker.read_text(encoding="ascii") == "untouched"


def test_unexpected_artifact_fails_closed_without_partial_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"

    def extra_artifact(*args: object, **kwargs: object) -> str:
        result = _fake_docker_cp(*args, **kwargs)  # type: ignore[arg-type]
        arguments = args[0]
        if arguments[-2].endswith("owner-gate-wheelhouse/."):
            (Path(arguments[-1]) / "unexpected.whl").write_bytes(b"unexpected")
        return result

    monkeypatch.setattr(harness, "_run", extra_artifact)
    monkeypatch.setattr(
        harness.package,
        "validate_wheelhouse",
        lambda **_kwargs: (),
    )

    with pytest.raises(
        harness.HarnessError,
        match="^owner_gate_debian12_wheelhouse_output_invalid$",
    ):
        harness._publish_verified_wheelhouse(
            container="fixed-container",
            destination=destination,
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".published.stage-*"))
