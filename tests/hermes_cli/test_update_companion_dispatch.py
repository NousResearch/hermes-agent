"""Both apply modes dispatch the selected native completion worklist."""
from types import SimpleNamespace

from hermes_cli import update_cmd_zip


def test_zip_dispatches_native_completion(tmp_path, monkeypatch):
    request = {"root": str(tmp_path)}
    calls = []
    monkeypatch.setattr(update_cmd_zip, "_abort_zip_update_if_dirty_tree", lambda: None)
    monkeypatch.setattr(update_cmd_zip, "_download_and_swap_zip", lambda *a: "a" * 40)
    monkeypatch.setattr("hermes_cli.update_cmd._complete_source_update", lambda r: calls.append(r.copy()))
    monkeypatch.setattr("hermes_cli.update_cmd._m", lambda: SimpleNamespace(
        PROJECT_ROOT=tmp_path, _resolve_update_branch=lambda args: "main"))
    update_cmd_zip._update_via_zip(SimpleNamespace(branch="main"), target_sha="a" * 40,
                                  target_repository="NousResearch/hermes-agent", completion_request=request)
    assert calls == [dict(request)]
    assert request["apply_mode"] == "zip"
