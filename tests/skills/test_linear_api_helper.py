from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_linear_api():
    script = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / "productivity"
        / "linear"
        / "scripts"
        / "linear_api.py"
    )
    spec = importlib.util.spec_from_file_location("linear_api_helper_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_accepts_documented_positional_variables(monkeypatch, capsys) -> None:
    linear_api = _load_linear_api()
    calls = []

    def fake_gql(query, variables=None):
        calls.append((query, variables))
        return {"ok": True}

    monkeypatch.setattr(linear_api, "gql", fake_gql)
    linear_api.main(["raw", "query($id: String!) { issue(id: $id) { id } }", '{"id":"ENG-42"}'])

    assert calls == [
        ("query($id: String!) { issue(id: $id) { id } }", {"id": "ENG-42"})
    ]
    assert '"ok": true' in capsys.readouterr().out


def test_raw_rejects_positional_and_flag_variables_together(capsys) -> None:
    linear_api = _load_linear_api()

    try:
        linear_api.main(["raw", "query { viewer { id } }", "{}", "--vars", "{}"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    assert "Use either positional variables_json or --vars" in capsys.readouterr().err
