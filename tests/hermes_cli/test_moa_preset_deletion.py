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

import errno
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
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


def test_deletion_survives_autosave_restart_and_backup_with_sparse_yaml(tmp_path, monkeypatch):
    """The real HTTP save keeps user YAML intact and never revives omitted names."""
    from starlette.testclient import TestClient

    from hermes_cli.config import load_config, read_raw_config
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    moa = _three_presets()
    moa.update(privacy_filter="full", save_traces=True, trace_dir="/custom/traces")
    moa["presets"]["keep_a"]["operator_note"] = None
    _seed(tmp_path, monkeypatch, moa)
    config_path = tmp_path / "config.yaml"
    untouched = (
        '# Operator rationale\n'
        'operator_settings:\n'
        '  keep: null  # explicit null, not an omitted key\n'
        '  label: "off"\n\n'
    )
    config_path.write_text(untouched + config_path.read_text(encoding="utf-8"), encoding="utf-8")
    # Backup filenames have second precision. Model a later edit explicitly so
    # this checks MoA fallback merging, not the upstream same-second collision
    # (which can leave ANY config setting's backup stale).
    from hermes_cli import config_backups

    with monkeypatch.context() as clock:
        clock.setattr(config_backups.time, "strftime", lambda _fmt: "20000101-000000")
        # Prime both caches before deleting; subsequent reads must see the save.
        assert "doomed" in read_raw_config()["moa"]["presets"]
        assert "doomed" in load_config()["moa"]["presets"]
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    survivors = {"keep_a", "keep_b"}
    response = client.put("/api/model/moa", json={
        "default_preset": "keep_a",
        "presets": {name: _preset("gpt-5.8").model_dump() for name in sorted(survivors)},
    })
    assert response.status_code == 200, response.text
    assert set(read_raw_config()["moa"]["presets"]) == survivors
    effective = client.get("/api/config")
    assert effective.status_code == 200, effective.text
    assert set(effective.json()["moa"]["presets"]) == survivors

    # Desktop settings autosave sends only edited keys, not its stale MoA snapshot.
    autosave = client.put("/api/config", json={"config": {"display": {"skin": "mono"}}})
    assert autosave.status_code == 200, autosave.text
    saved = _on_disk(tmp_path)
    assert set(saved["moa"]["presets"]) == survivors
    assert saved["operator_settings"] == {"keep": None, "label": "off"}
    assert saved["moa"]["presets"]["keep_a"]["operator_note"] is None
    assert saved["moa"]["privacy_filter"] == "full"
    assert saved["moa"]["save_traces"] is True
    assert saved["moa"]["trace_dir"] == "/custom/traces"
    assert saved["display"]["skin"] == "mono"
    assert "terminal" not in saved, "effective defaults leaked into sparse YAML"
    text = config_path.read_text(encoding="utf-8")
    # The round-trip writer may spell YAML null as an empty value; preserve
    # its semantic value (asserted above), comments, quoting and key order.
    assert text.startswith("# Operator rationale\noperator_settings:\n")
    assert "# explicit null, not an omitted key" in text
    assert '  label: "off"\n\n' in text
    assert list(saved)[:2] == ["operator_settings", "moa"]

    # Fresh imports use the actual saved YAML, then the persisted last-good backup.
    probe = (
        "import json; from hermes_cli.config import load_config; "
        "print(json.dumps(sorted(load_config()['moa']['presets'])))"
    )
    for broken in (False, True):
        if broken:
            config_path.write_text("moa: [unterminated\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-B", "-c", probe],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "HERMES_HOME": str(tmp_path)},
            capture_output=True, text=True, check=True, timeout=30,
        )
        assert json.loads(result.stdout.splitlines()[-1]) == sorted(survivors)
        if broken:
            assert set(load_config()["moa"]["presets"]) == survivors


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


@pytest.mark.parametrize("legacy", [False, True], ids=["named", "legacy-flat"])
def test_moa_save_preserves_a_concurrent_partial_config_write(tmp_path, monkeypatch, legacy):
    """A writer outside the dashboard lock must not lose its update to a stale MoA read."""
    from concurrent.futures import ThreadPoolExecutor

    from starlette.testclient import TestClient

    from hermes_cli import config as config_mod
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    _seed(tmp_path, monkeypatch, _three_presets(), display={"skin": "slate"})
    read_complete, resume_read = threading.Event(), threading.Event()
    real_read = config_mod.require_readable_config_before_write

    def gated_read(*args, **kwargs):
        raw = real_read(*args, **kwargs)
        if not read_complete.is_set():
            read_complete.set()
            assert resume_read.wait(30), "concurrent writer never reached the config lock"
        return raw

    def write_other_setting():
        assert read_complete.wait(30), "MoA save never read the config"
        # If MoA owns the RMW lock, let it finish before this writer proceeds.
        # Otherwise land this update inside its read/save gap, without a timing race.
        acquired = config_mod._CONFIG_LOCK.acquire(blocking=False)
        try:
            if not acquired:
                resume_read.set()
            config_mod.save_config({"display": {"skin": "mono"}}, merge_existing=True)
        finally:
            if acquired:
                config_mod._CONFIG_LOCK.release()
            resume_read.set()

    monkeypatch.setattr(config_mod, "require_readable_config_before_write", gated_read)
    payload = (
        _preset("gpt-5.8").model_dump()
        if legacy else
        {"default_preset": "keep_a", "presets": {"keep_a": _preset("gpt-5.8").model_dump()}}
    )
    client = TestClient(app)
    with ThreadPoolExecutor(max_workers=2) as pool:
        moa_save = pool.submit(
            client.put, "/api/model/moa", json=payload,
            headers={_SESSION_HEADER_NAME: _SESSION_TOKEN},
        )
        other_save = pool.submit(write_other_setting)
        try:
            response = moa_save.result(timeout=60)
            other_save.result(timeout=60)
        finally:
            resume_read.set()
    assert response.status_code == 200, response.text
    saved = _on_disk(tmp_path)
    assert saved["display"]["skin"] == "mono"
    assert set(saved["moa"]["presets"]) == ({"keep_a", "doomed", "keep_b"} if legacy else {"keep_a"})
    assert saved["moa"]["presets"]["keep_a"]["reference_models"][0]["model"] == "gpt-5.8"


@pytest.mark.parametrize("legacy", [False, True], ids=["named", "legacy-flat"])
def test_moa_save_refuses_transient_read_failure_without_losing_config(tmp_path, monkeypatch, legacy):
    """A recovered second read must not turn the first read's fallback into a full write."""
    from starlette.testclient import TestClient

    from hermes_cli import config as config_mod
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    _seed(tmp_path, monkeypatch, _three_presets(), operator_settings={"keep": None})
    config_path = tmp_path / "config.yaml"
    before = config_path.read_bytes()
    real_load = config_mod.fast_safe_load
    failed_reads = []

    def fail_once(stream):
        if getattr(stream, "name", None) == str(config_path) and not failed_reads:
            failed_reads.append(str(config_path))
            raise OSError(errno.EMFILE, "Too many open files")
        return real_load(stream)

    monkeypatch.setattr(config_mod, "fast_safe_load", fail_once)
    payload = (
        _preset("gpt-5.8").model_dump()
        if legacy else
        {"default_preset": "keep_a", "presets": {"keep_a": _preset("gpt-5.8").model_dump()}}
    )
    client = TestClient(app)
    headers = {_SESSION_HEADER_NAME: _SESSION_TOKEN}
    refused = client.put("/api/model/moa", json=payload, headers=headers)
    assert failed_reads == [str(config_path)]
    assert refused.status_code == 500, refused.text
    assert config_path.read_bytes() == before

    retry = client.put("/api/model/moa", json=payload, headers=headers)
    assert retry.status_code == 200, retry.text
    saved = _on_disk(tmp_path)
    assert saved["operator_settings"] == {"keep": None}
    assert set(saved["moa"]["presets"]) == ({"keep_a", "doomed", "keep_b"} if legacy else {"keep_a"})
    assert saved["moa"]["presets"]["keep_a"]["reference_models"][0]["model"] == "gpt-5.8"


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


@pytest.mark.parametrize("include_presets", [False, True], ids=["missing-presets", "empty-presets"])
def test_legacy_flat_update_preserves_named_presets_and_valid_default(
    tmp_path, monkeypatch, include_presets
):
    """Older flat clients preserve named presets and a valid persisted active selection."""
    home = tmp_path / ".hermes"
    home.mkdir()
    seeded = _three_presets()
    seeded["active_preset"] = "keep_b"
    _seed(home, monkeypatch, seeded)

    payload = {
        "active_preset": "",
        "reference_models": [MoaModelSlot(provider="openai-codex", model="gpt-5.8")],
        "aggregator": MoaModelSlot(provider="anthropic", model="claude-opus-5"),
        "enabled": True,
    }
    if include_presets:
        payload["presets"] = {}

    set_moa_models(MoaConfigPayload(**payload))

    moa = _on_disk(home)["moa"]
    assert set(moa["presets"]) == {"keep_a", "doomed", "keep_b"}
    assert moa["default_preset"] == "keep_a"
    assert moa["active_preset"] == "keep_b"
    assert moa["active_preset"] in moa["presets"]
    assert "default" not in moa["presets"]
    assert moa["presets"]["keep_a"]["reference_models"] == [
        {"provider": "openai-codex", "model": "gpt-5.8", "enabled": True}
    ]
    assert moa["presets"]["keep_a"]["aggregator"] == {
        "provider": "anthropic", "model": "claude-opus-5"
    }
    # Legacy flat fields represent the saved default preset. Runtime resolution uses
    # default_preset when no preset name is explicitly requested; active_preset is metadata.
    from hermes_cli.moa_config import resolve_moa_preset

    assert resolve_moa_preset(moa)["reference_models"] == moa["presets"]["keep_a"]["reference_models"]
    assert moa["presets"]["keep_b"] == seeded["presets"]["keep_b"]
    assert moa["presets"]["doomed"] == seeded["presets"]["doomed"]


def test_legacy_flat_update_with_named_default_and_empty_map_does_not_fail(tmp_path, monkeypatch):
    """A legacy flat payload's default_preset field cannot rename its synthesized map."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _seed(home, monkeypatch, _three_presets())

    result = set_moa_models(
        MoaConfigPayload(
            default_preset="keep_b",
            presets={},
            reference_models=[MoaModelSlot(provider="openai-codex", model="gpt-5.8")],
            aggregator=MoaModelSlot(provider="anthropic", model="claude-opus-5"),
            enabled=True,
        )
    )

    moa = _on_disk(home)["moa"]
    assert result["ok"] is True
    assert set(moa["presets"]) == {"keep_a", "doomed", "keep_b"}
    assert moa["default_preset"] == "keep_a"
    assert moa["presets"]["keep_a"]["reference_models"] == [
        {"provider": "openai-codex", "model": "gpt-5.8", "enabled": True}
    ]
