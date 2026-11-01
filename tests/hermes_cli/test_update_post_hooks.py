"""``updates.post_hooks`` -- post-dependency-sync overlay verifier (#115667).

``hermes update`` reinstalls venv deps with no post-install hook, so operator
overlays inside site-packages (e.g. a patched third-party transport) vanish
silently. Operators declare ``updates.post_hooks`` as path + marker pairs;
after the sync a verifier re-checks each marker, re-runs the hook's command
when one is declared, and logs the outcome. The verifier never raises -- the
update must finish even when a hook is broken.
"""

import inspect

import hermes_cli.config as hermes_config
import hermes_cli.update_post_hooks as deps


def _cfg(monkeypatch, updates):
    monkeypatch.setattr(
        hermes_config, "load_config", lambda *a, **kw: {"updates": updates})


def test_no_hooks_configured_is_silent_noop(monkeypatch, tmp_path, capsys):
    _cfg(monkeypatch, {})
    assert deps._verify_update_post_hooks(project_root=tmp_path) == []
    assert capsys.readouterr().out == ""


def test_marker_present_verifies(monkeypatch, tmp_path, capsys):
    target = tmp_path / "transport.py"
    target.write_text("proxies=none  # operator overlay\n")
    _cfg(monkeypatch, {"post_hooks": [
        {"path": str(target), "marker": "operator overlay"}]})
    results = deps._verify_update_post_hooks(project_root=tmp_path)
    assert [r["status"] for r in results] == ["verified"]
    out = capsys.readouterr().out
    assert "verified" in out and "transport.py" in out


def test_missing_marker_without_command_warns_loudly(monkeypatch, tmp_path, capsys):
    target = tmp_path / "transport.py"
    target.write_text("# pristine upstream file\n")
    _cfg(monkeypatch, {"post_hooks": [
        {"path": str(target), "marker": "OPERATOR-OVERLAY"}]})
    results = deps._verify_update_post_hooks(project_root=tmp_path)
    assert [r["status"] for r in results] == ["failed"]
    out = capsys.readouterr().out
    assert "OPERATOR-OVERLAY" in out


def test_missing_marker_with_command_reapplies(monkeypatch, tmp_path, capsys):
    target = tmp_path / "transport.py"
    target.write_text("# pristine upstream file\n")
    _cfg(monkeypatch, {"post_hooks": [{
        "path": str(target),
        "marker": "OPERATOR-OVERLAY",
        "command": "echo OPERATOR-OVERLAY >> " + str(target),
    }]})
    results = deps._verify_update_post_hooks(project_root=tmp_path)
    assert [r["status"] for r in results] == ["reapplied"]
    assert "OPERATOR-OVERLAY" in target.read_text()
    assert "re-applied" in capsys.readouterr().out


def test_missing_file_with_command_creates_it(monkeypatch, tmp_path, capsys):
    target = tmp_path / "transport.py"
    _cfg(monkeypatch, {"post_hooks": [{
        "path": str(target),
        "marker": "OPERATOR-OVERLAY",
        "command": "echo OPERATOR-OVERLAY > " + str(target),
    }]})
    results = deps._verify_update_post_hooks(project_root=tmp_path)
    assert [r["status"] for r in results] == ["reapplied"]
    assert "OPERATOR-OVERLAY" in target.read_text()
    capsys.readouterr()


def test_failing_command_reports_failed_not_raise(monkeypatch, tmp_path, capsys):
    target = tmp_path / "transport.py"
    target.write_text("# pristine upstream file\n")
    _cfg(monkeypatch, {"post_hooks": [{
        "path": str(target), "marker": "OPERATOR-OVERLAY",
        "command": "exit 3",
    }]})
    results = deps._verify_update_post_hooks(project_root=tmp_path)
    assert [r["status"] for r in results] == ["failed"]
    assert "FAILED" in capsys.readouterr().out


def test_malformed_config_is_skipped_silently(monkeypatch, tmp_path, capsys):
    for bad in ("notalist", None, 42,
                [{"path": "x.py"}], [{"marker": "M"}],
                ["astring", 42, None, {"path": "", "marker": ""}]):
        _cfg(monkeypatch, {"post_hooks": bad})
        assert deps._verify_update_post_hooks(project_root=tmp_path) == []
    assert capsys.readouterr().out == ""


def test_relative_path_resolves_inside_venv_site_packages(monkeypatch, tmp_path):
    sp = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "pkg"
    sp.mkdir(parents=True)
    monkeypatch.setattr(deps, "project_venv_dir", lambda root: root / "venv")
    (sp / "transport.py").write_text("# OPERATOR-OVERLAY\n")
    _cfg(monkeypatch, {"post_hooks": [
        {"path": "pkg/transport.py", "marker": "OPERATOR-OVERLAY"}]})
    results = deps._verify_update_post_hooks(project_root=tmp_path)
    assert [r["status"] for r in results] == ["verified"]
    assert "site-packages" in results[0]["path"]


def test_config_load_failure_never_raises(monkeypatch, tmp_path, capsys):
    def _boom(*a, **kw):
        raise RuntimeError("config unreadable")
    monkeypatch.setattr(hermes_config, "load_config", _boom)
    assert deps._verify_update_post_hooks(project_root=tmp_path) == []
    assert capsys.readouterr().out == ""


def test_update_tail_wires_the_post_hook_verifier():
    """The verifier must run after the update tail, so the verdict reflects the
    final venv state (a mid-update reinstall can wipe a file an earlier phase
    already reported OK -- #115667)."""
    from hermes_cli import update_completion

    source = inspect.getsource(update_completion._complete_selected)
    assert "_verify_update_post_hooks(project_root=root)" in source
