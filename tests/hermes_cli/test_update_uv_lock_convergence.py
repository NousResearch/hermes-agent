"""Lock convergence for source-only ``hermes update`` pulls."""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import update_cmd, update_cmd_deps


def _run_unchanged_dependency_sync(
    tmp_path,
    monkeypatch,
    *,
    uv_bin="uv",
    has_lock=True,
    uv_managed=True,
    is_termux=False,
    sync_returncode=0,
):
    root = tmp_path / "checkout"
    venv = root / "venv"
    venv.mkdir(parents=True)
    config = "uv = 0.8.0\n" if uv_managed else "home = /usr/bin\n"
    (venv / "pyvenv.cfg").write_text(config, encoding="utf-8")
    if has_lock:
        (root / "uv.lock").write_text("version = 1\nrevision = 3\n", encoding="utf-8")

    verified = []
    facade = SimpleNamespace(
        PROJECT_ROOT=root,
        _abort_dependency_sync_if_self_locked=lambda _resume: None,
        _verify_core_dependencies_installed=lambda *_args, **_kwargs: verified.append("core"),
        _verify_console_scripts_installed=lambda *_args, **_kwargs: verified.append("scripts"),
        _clear_update_incomplete_marker=lambda: None,
        _reload_updated_runtime_modules=lambda: None,
        _upgrade_pip_before_lazy_refresh=lambda *_args, **_kwargs: None,
        _refresh_active_lazy_features=lambda *_args, **_kwargs: True,
        _clear_lazy_refresh_incomplete_marker=lambda: None,
        _restore_active_tool_dependencies=lambda *_args, **_kwargs: None,
        _refresh_active_memory_provider_dependencies=lambda: None,
        _is_termux_env=lambda _env: is_termux,
    )

    monkeypatch.setattr(update_cmd_deps, "_refuse_update_if_venv_foreign_owned", lambda _root: None)
    monkeypatch.setattr(update_cmd_deps, "_editable_install_is_current", lambda *_args: True)
    monkeypatch.setattr(update_cmd_deps, "_ensure_uv_for_termux", lambda _pip: None)
    monkeypatch.setattr(update_cmd, "_m", lambda: facade)
    monkeypatch.setattr(update_cmd, "_pip_install_prefix", lambda _uv: (["uv", "pip"], {}))
    monkeypatch.setattr(update_cmd, "_sweep_bytecode_after_update", lambda _branch: None)
    monkeypatch.setattr(update_cmd, "_write_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(update_cmd, "_write_lazy_refresh_incomplete_marker", lambda: None)
    monkeypatch.setattr(
        update_cmd, "_validate_critical_modules_import", lambda _root: (True, None, None)
    )

    with (
        patch("hermes_cli.managed_uv.update_managed_uv"),
        patch("hermes_cli.managed_uv.ensure_uv", return_value=uv_bin),
        patch("hermes_cli.update_cmd_deps.subprocess.run") as run,
    ):
        run.return_value.returncode = sync_returncode
        update_cmd_deps._sync_python_dependencies_after_pull(
            ["git"],
            "main",
            "before",
            active_lazy_features=[],
            active_tool_dependencies=[],
            _windows_gateway_resume=None,
        )

    return root, run, verified


def test_source_only_update_syncs_stale_transitives_to_uv_lock(tmp_path, monkeypatch):
    root, run, verified = _run_unchanged_dependency_sync(tmp_path, monkeypatch)

    run.assert_called_once_with(
        ["uv", "sync", "--extra", "all", "--locked"],
        cwd=root,
        env={"UV_PROJECT_ENVIRONMENT": str(root / "venv")},
        check=False,
    )
    assert verified == ["core", "scripts"]


def test_locked_sync_controls_fail_open_to_legacy_verification(tmp_path, monkeypatch):
    cases = (
        {"has_lock": False},
        {"uv_bin": None},
        {"uv_managed": False},
        {"is_termux": True},
        {"sync_returncode": 1},
    )

    for index, options in enumerate(cases):
        _, run, verified = _run_unchanged_dependency_sync(
            tmp_path / str(index), monkeypatch, **options
        )

        sync_calls = [call for call in run.call_args_list if call.args[0][:2] == ["uv", "sync"]]
        assert len(sync_calls) == (1 if options.get("sync_returncode") else 0)
        assert verified == ["core", "scripts"]
