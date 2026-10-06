from __future__ import annotations

import json
import runpy
from pathlib import Path


def test_generate_config_schema_outputs_valid_json(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "generate_config_schema.py"
    output_path = repo_root / "website" / "static" / "schemas" / "hermes-config.schema.json"

    before = output_path.read_text(encoding="utf-8") if output_path.exists() else None
    try:
        monkeypatch.chdir(repo_root)
        runpy.run_path(str(script_path), run_name="__main__")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert payload["title"] == "Hermes Agent config.yaml"
        assert "model" in payload["properties"]
        assert "custom_providers" in payload["properties"]
        assert "platform_toolsets" in payload["properties"]
    finally:
        if before is None:
            output_path.unlink(missing_ok=True)
        else:
            output_path.write_text(before, encoding="utf-8")
