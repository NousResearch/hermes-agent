from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import models as model_catalog


def _load_refresh_module():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "refresh_opencode_free_catalog.py"
    spec = importlib.util.spec_from_file_location("refresh_opencode_free_catalog", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_refresh_opencode_free_catalog_main_json(monkeypatch, capsys):
    monkeypatch.setattr(
        model_catalog,
        "opencode_free_model_ids",
        lambda **kwargs: ["big-pickle", "qwen3.6-plus-free"],
    )
    monkeypatch.setattr(
        model_catalog,
        "resolve_config_model_id",
        lambda provider, model_id, **kwargs: "big-pickle",
    )

    refresh_script = _load_refresh_module()
    monkeypatch.setattr(
        refresh_script,
        "argparse",
        SimpleNamespace(
            ArgumentParser=lambda description=None: SimpleNamespace(
                add_argument=lambda *a, **k: None,
                parse_args=lambda: SimpleNamespace(json=True, force=True),
            )
        ),
    )

    assert refresh_script.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["primary_model"] == "big-pickle"
    assert payload["free_models"] == ["big-pickle", "qwen3.6-plus-free"]
