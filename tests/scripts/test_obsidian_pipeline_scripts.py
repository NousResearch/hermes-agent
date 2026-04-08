import importlib.util
import json
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SYNC_SCRIPT = ROOT / "scripts" / "ops" / "obsidian_canonical_sync.py"
HEALTH_SCRIPT = ROOT / "scripts" / "ops" / "obsidian_pipeline_healthcheck.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_obsidian_canonical_sync_ledger_is_idempotent(tmp_path, monkeypatch):
    mod = _load_module(SYNC_SCRIPT, "obsidian_canonical_sync_test")

    sent = []

    def fake_put(api_base, api_key, note_path, markdown, timeout=20):
        sent.append((note_path, markdown))

    monkeypatch.setattr(mod, "_put_markdown", fake_put)

    items = [{"bookmark_id": "123", "title": "hello", "url": "https://x.test/1", "content": "abc"}]
    ledger = tmp_path / "ledger.json"

    first = mod.sync_bookmarks(
        items,
        api_base="https://127.0.0.1:27124",
        api_key="k",
        note_root="Siftly/Bookmarks",
        ledger_path=ledger,
        dry_run=False,
    )
    second = mod.sync_bookmarks(
        items,
        api_base="https://127.0.0.1:27124",
        api_key="k",
        note_root="Siftly/Bookmarks",
        ledger_path=ledger,
        dry_run=False,
    )

    assert first["written"] == 1
    assert second["written"] == 0
    assert second["skipped"] == 1
    assert len(sent) == 1
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert "123" in saved


def test_obsidian_pipeline_healthcheck_repair_creates_ledger(tmp_path, capsys):
    ledger_path = tmp_path / "state" / "ledger.json"

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(HEALTH_SCRIPT),
            "--api-base",
            "https://127.0.0.1:65534",
            "--ledger",
            str(ledger_path),
            "--repair",
        ]
        try:
            runpy.run_path(str(HEALTH_SCRIPT), run_name="__main__")
        except SystemExit as exc:
            code = int(exc.code or 0)
        out = capsys.readouterr().out
    finally:
        sys.argv = old_argv

    payload = json.loads(out)
    assert ledger_path.exists()
    assert payload["repair_applied"] is True
    assert payload["ledger_exists"] is True
    assert code == 1  # no key configured, so health should fail
