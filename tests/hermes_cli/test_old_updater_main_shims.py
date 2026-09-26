"""Historical main imports must not restart pre-PM updater work after a swap."""

from pathlib import Path

import pytest

from tests.compat.old_updater_support import (
    no_external_work as no_external_work,
)


@pytest.fixture
def historical_main(no_external_work):
    from hermes_cli import main

    return main


def test_historical_main_data_and_skipped_probes_preserve_caller_shapes(historical_main, tmp_path):
    main = historical_main
    from hermes_cli import main_web_build

    # Old recorders compose this name with PROJECT_ROOT. It remains data only.
    assert tmp_path / main._BYTECODE_FINGERPRINT_FILE == (
        tmp_path / main_web_build._BYTECODE_FINGERPRINT_FILE
    )
    failed = ["hermes.exe"]
    try:
        raise main.ShimQuarantineError(failed)
    except main.ShimQuarantineError as exc:
        assert isinstance(exc, RuntimeError)
        assert exc.failed_shims == failed
        assert exc.failed_shims is not failed
        assert failed[0] in str(exc)

    before_files = set(tmp_path.rglob("*"))
    prefix = ["uv", "pip"]
    env = {"VIRTUAL_ENV": str(tmp_path)}
    # None means indeterminate to the historical repair caller, NOT healthy [].
    assert main._detect_broken_lazy_refresh_imports(prefix, env=env) is None
    assert main._resolve_install_target_python(prefix, env) is None
    moved = [(tmp_path / "hermes.exe", tmp_path / "hermes.exe.old")]
    before = list(moved)
    assert main._restore_quarantined_exes(moved) is None
    assert moved == before
    assert main._write_web_ui_build_stamp(tmp_path, tmp_path / "web") is None
    assert prefix == ["uv", "pip"]
    assert env == {"VIRTUAL_ENV": str(tmp_path)}
    assert set(tmp_path.rglob("*")) == before_files


def test_historical_marker_cleanup_preserves_path_and_is_idempotent(historical_main, tmp_path, monkeypatch):
    # The retired writer now hands off. Cleanup of an existing legacy marker
    # remains supported; PM's recovery lifecycle is covered in test_early_recovery.
    monkeypatch.setattr(historical_main, "PROJECT_ROOT", tmp_path)
    marker = historical_main._update_marker_path()
    assert marker == tmp_path / ".update-incomplete"
    marker.write_text("started=1\npid=0\n", encoding="utf-8")
    assert historical_main._clear_update_incomplete_marker() is None
    assert not marker.exists()
    assert historical_main._clear_update_incomplete_marker() is None
