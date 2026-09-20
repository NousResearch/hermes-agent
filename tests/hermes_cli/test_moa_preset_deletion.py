"""A MoA preset dropped from the payload must leave config.yaml.

``PUT /api/model/moa`` expresses a deletion by OMITTING the preset from
``presets``. The save used ``save_config(..., merge_existing=True)``, whose
``_deep_merge`` recurses dict-over-dict and therefore restored every omitted
preset from disk — the GUI's delete button returned ok while the preset stayed
in config.yaml forever. ``presets`` is authoritative on write; undeclared
sibling keys (``save_traces``, ``trace_dir``) must still survive (#58819), and
other config sections must not be clobbered (#89184).

Real config pipeline against a temp HERMES_HOME — a mocked save cannot show the
merge that caused the bug.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from hermes_cli.web_models import MoaConfigPayload, MoaModelSlot, MoaPresetPayload
from hermes_cli.web_routers.models import set_moa_models


def _preset(model: str) -> MoaPresetPayload:
    return MoaPresetPayload(
        reference_models=[MoaModelSlot(provider="openai-codex", model=model)],
        aggregator=MoaModelSlot(provider="anthropic", model="claude-opus-5"),
        enabled=True,
    )


def _on_disk(home) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


def _seed(home, monkeypatch, moa: dict, **sections) -> None:
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"moa": moa, **sections}), encoding="utf-8")


def _three_presets() -> dict:
    return {
        "default_preset": "keep_a",
        "presets": {
            "keep_a": {
                "reference_models": [{"provider": "openai-codex", "model": "gpt-5.5"}],
                "aggregator": {"provider": "anthropic", "model": "claude-opus-5"},
                "enabled": True,
            },
            "doomed": {
                "reference_models": [{"provider": "openai-codex", "model": "gpt-5.6-luna"}],
                "aggregator": {"provider": "anthropic", "model": "claude-opus-5"},
                "enabled": True,
            },
            "keep_b": {
                "reference_models": [{"provider": "openai-codex", "model": "gpt-5.7"}],
                "aggregator": {"provider": "anthropic", "model": "claude-opus-5"},
                "enabled": True,
            },
        },
    }


def test_omitted_preset_is_deleted_from_disk(tmp_path, monkeypatch):
    """The whole point: a preset the payload omits must not survive the save."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _seed(home, monkeypatch, _three_presets())

    # The GUI re-sends the surviving presets; "doomed" is simply absent.
    set_moa_models(
        MoaConfigPayload(
            default_preset="keep_a",
            active_preset="",
            presets={"keep_a": _preset("gpt-5.5"), "keep_b": _preset("gpt-5.7")},
        )
    )

    presets = _on_disk(home)["moa"]["presets"]
    assert "doomed" not in presets, "deleted preset was restored by the merge"
    assert set(presets) == {"keep_a", "keep_b"}


def test_deletion_preserves_undeclared_moa_keys_and_other_sections(tmp_path, monkeypatch):
    """Authoritative presets must not cost us #58819 (save_traces) or #89184 (siblings)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    moa = _three_presets()
    moa.update(save_traces=True, trace_dir="/custom/traces")
    chain = [{"provider": "custom", "model": "glm-5.08", "base_url": "http://gw:8080/v1"}]
    _seed(home, monkeypatch, moa, fallback_providers=chain)

    set_moa_models(
        MoaConfigPayload(
            default_preset="keep_a",
            active_preset="",
            presets={"keep_a": _preset("gpt-5.5"), "keep_b": _preset("gpt-5.7")},
        )
    )

    on_disk = _on_disk(home)
    assert "doomed" not in on_disk["moa"]["presets"]
    assert on_disk["moa"]["save_traces"] is True, "save_traces dropped (#58819)"
    assert on_disk["moa"]["trace_dir"] == "/custom/traces", "trace_dir dropped (#58819)"
    assert on_disk["fallback_providers"] == chain, "MoA save clobbered fallback_providers (#89184)"


def test_adding_a_preset_still_works(tmp_path, monkeypatch):
    """Authoritative writes must not break the add/edit path."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _seed(home, monkeypatch, _three_presets())

    set_moa_models(
        MoaConfigPayload(
            default_preset="keep_a",
            active_preset="",
            presets={
                "keep_a": _preset("gpt-5.5"),
                "doomed": _preset("gpt-5.6-luna"),
                "keep_b": _preset("gpt-5.7"),
                "fresh": _preset("gpt-5.9"),
            },
        )
    )

    presets = _on_disk(home)["moa"]["presets"]
    assert set(presets) == {"keep_a", "doomed", "keep_b", "fresh"}
    assert presets["fresh"]["reference_models"][0]["model"] == "gpt-5.9"


def test_named_map_does_not_receive_schema_default_on_load(tmp_path, monkeypatch):
    """An explicit named map owns its names; loader defaults must not add ``default``."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _seed(home, monkeypatch, _three_presets())

    from hermes_cli.config import load_config

    loaded = load_config()
    assert set(loaded["moa"]["presets"]) == {"keep_a", "doomed", "keep_b"}


def test_named_map_stays_authoritative_in_a_fresh_process_and_backup_fallback(tmp_path, monkeypatch):
    """A restart and last-known-good fallback must not reinsert the schema ``default``."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _seed(home, monkeypatch, _three_presets())

    from hermes_cli.config import load_config

    # The first valid load creates the real ``good`` backup used by a fresh process.
    assert set(load_config()["moa"]["presets"]) == {"keep_a", "doomed", "keep_b"}
    (home / "config.yaml").write_text("moa: [unterminated\n", encoding="utf-8")

    probe = (
        "import json; from hermes_cli.config import load_config; "
        "print(json.dumps(sorted(load_config()['moa']['presets'])))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=__import__("pathlib").Path(__file__).resolve().parents[2],
        env={**os.environ, "HERMES_HOME": str(home)},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == ["doomed", "keep_a", "keep_b"]


def test_missing_moa_map_still_receives_schema_defaults(tmp_path, monkeypatch):
    """Without a user MoA map, the built-in default remains available."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")

    from hermes_cli.config import DEFAULT_CONFIG, load_config
    from hermes_cli.moa_config import normalize_moa_config

    actual = normalize_moa_config(load_config()["moa"])
    expected = normalize_moa_config(DEFAULT_CONFIG["moa"])
    assert actual == expected


def test_deletion_preserves_privacy_filter_and_retained_preset_metadata(tmp_path, monkeypatch):
    """A MoA edit must not reset an undeclared privacy policy or metadata."""
    home = tmp_path / ".hermes"
    home.mkdir()
    moa = _three_presets()
    moa.update(
        privacy_filter="full",
        save_traces=True,
        trace_dir="/custom/traces",
        presets={**moa["presets"], "keep_a": {**moa["presets"]["keep_a"], "operator_note": "retain me"}},
    )
    _seed(home, monkeypatch, moa)

    set_moa_models(
        MoaConfigPayload(
            default_preset="keep_a",
            active_preset="",
            presets={"keep_a": _preset("gpt-5.5"), "keep_b": _preset("gpt-5.7")},
        )
    )

    on_disk = _on_disk(home)["moa"]
    assert "doomed" not in on_disk["presets"]
    assert on_disk["privacy_filter"] == "full"
    assert on_disk["save_traces"] is True
    assert on_disk["trace_dir"] == "/custom/traces"
    assert on_disk["presets"]["keep_a"]["operator_note"] == "retain me"
