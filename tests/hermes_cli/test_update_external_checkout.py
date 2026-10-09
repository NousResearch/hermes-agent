"""Regression coverage for separate-checkout gateways (#117449)."""

from pathlib import Path
from types import SimpleNamespace

from hermes_cli import update_cmd_fleet, update_receipt


def _checkout(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    module = root / "hermes_cli" / "main.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    return root


def test_different_checkout_is_external_but_same_checkout_sha_skew_is_stale(tmp_path):
    updated = _checkout(tmp_path, "updated")
    separate = _checkout(tmp_path, "separate")

    external = update_receipt._fleet_row(
        "pinned", 1, "old", "1", "new",
        code_root=separate, expected_root=updated, served_profiles=["pinned", "coder", "coder"],
    )
    stale = update_receipt._fleet_row(
        "default", 2, "old", "1", "new",
        code_root=updated, expected_root=updated,
    )

    assert external["state"] == "external"
    assert external["served_profiles"] == ["pinned", "coder"]
    assert stale["state"] == "stale"


def test_unresolved_checkout_fails_closed_to_sha_comparison():
    row = update_receipt._fleet_row("default", 1, "old", "1", "new")
    assert row["state"] == "stale"


def test_code_root_requires_absolute_path(tmp_path, monkeypatch):
    _checkout(tmp_path, "updated")
    monkeypatch.chdir(tmp_path / "updated")
    assert update_receipt._code_root_for_path("-m") is None
    assert update_receipt._code_root_for_path("hermes_cli/main.py") is None


def test_gateway_root_uses_pid_guarded_status_argv(tmp_path, monkeypatch):
    root = _checkout(tmp_path, "pinned")
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda _path: {"pid": 42, "argv": [str(root / "hermes_cli" / "main.py")]},
    )
    assert update_receipt._gateway_code_root(42, tmp_path / "home") == root.resolve()


def test_gateway_root_uses_virtualenv_when_module_path_is_absent(tmp_path, monkeypatch):
    root = _checkout(tmp_path, "pinned")
    venv = root / "venv"
    monkeypatch.setattr("gateway.status.read_runtime_status", lambda _path: {"pid": 42, "argv": ["-m"]})
    fake_process = SimpleNamespace(
        environ=lambda: {"VIRTUAL_ENV": str(venv)},
        exe=lambda: "/usr/bin/python",
        cmdline=lambda: ["python", "-m", "hermes_cli.main"],
    )
    monkeypatch.setattr("psutil.Process", lambda _pid: fake_process)
    assert update_receipt._gateway_code_root(42, tmp_path / "home") == root.resolve()


def test_external_receipt_row_does_not_report_runtime_skew():
    receipt = {"fleet": [{"profile": "pinned", "state": "external", "code_sha": "foreign"}]}
    assert update_cmd_fleet._receipt_reports_stale_runtime(receipt, "updated") is False


def test_external_gateway_does_not_block_pending_restart_discharge(monkeypatch):
    rows = [
        {"profile": "default", "state": "current", "code_sha": "updated"},
        {
            "profile": "pinned", "state": "external", "code_sha": "foreign",
            "served_profiles": ["pinned", "coder"],
        },
    ]
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda: rows)
    owed = {("gateway", "default"), ("gateway", "pinned"), ("gateway", "coder")}
    assert update_cmd_fleet._live_fleet_covers_receipt("updated", {}, owed) is True


def test_external_gateway_does_not_fail_matrix(capsys):
    failed = update_receipt.print_fleet_version_matrix([
        {
            "profile": "pinned", "pid": 42, "code_sha": "foreign",
            "state": "external", "code_root": "/srv/hermes-pinned",
        }
    ])
    output = capsys.readouterr().out
    assert failed is False
    assert "separate checkout" in output
    assert "/srv/hermes-pinned" in output


def test_collect_fleet_versions_classifies_separate_checkout_gateway(tmp_path, monkeypatch):
    """The production entry point resolves the gateway's code root and reports ``external``.

    A pid-guarded ``gateway_state.json`` whose argv points at a pinned checkout must
    classify as ``external`` (never ``stale``) even though its sha is foreign.
    """
    import json

    pinned = _checkout(tmp_path, "pinned")
    home = tmp_path / "fleet_home"
    home.mkdir()
    (home / "gateway_state.json").write_text(json.dumps({
        "gateway_state": "running", "kind": "hermes-gateway", "pid": 4242,
        "argv": [str(pinned / "hermes_cli" / "main.py"), "gateway", "run"],
        "code_sha": "f" * 40, "code_version": "0.0.1",
    }), encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": "a" * 40, "short_sha": "a" * 8, "version": "1.0", "source": "git"},
    )
    monkeypatch.setattr("profiles.paths._get_default_hermes_home", lambda: home)
    monkeypatch.setattr("profiles.paths._get_profiles_root", lambda: tmp_path / "no_profiles")
    monkeypatch.setattr(update_receipt, "_socket_identity", lambda _home: None)
    monkeypatch.setattr("gateway.status.live_gateway_pid_for_home", lambda _home: 4242)

    fleet = update_receipt.collect_fleet_versions()

    assert [row["state"] for row in fleet] == ["external"]
    assert fleet[0]["code_root"] == str(pinned.resolve())
    assert update_receipt.print_fleet_version_matrix(fleet) is False  # matrix does not fail the update
