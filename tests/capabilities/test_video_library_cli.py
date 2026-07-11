import json

from capabilities.video_library import cli


def test_cli_scan_prints_machine_readable_json(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "scan_library",
        lambda library_id, dry_run=False: {
            "complete": 4,
            "dry_run": dry_run,
            "library_id": library_id,
        },
    )

    assert cli.main(["scan", "--library", "beef-noodle"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["library_id"] == "beef-noodle"
    assert payload["complete"] == 4


def test_cli_dry_run_forwards_without_writes(monkeypatch, capsys):
    calls = []

    def fake_scan(library_id, dry_run=False):
        calls.append((library_id, dry_run))
        return {"dry_run": dry_run, "library_id": library_id, "writes_planned": []}

    monkeypatch.setattr(cli, "scan_library", fake_scan)

    assert cli.main(["scan", "--library", "beef-noodle", "--dry-run"]) == 0

    assert calls == [("beef-noodle", True)]
    assert json.loads(capsys.readouterr().out)["writes_planned"] == []


def test_cli_search_prints_results(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "search_library",
        lambda library_id, query, tag="", limit=50: {
            "clips": [{"description": "厨师手工拉面", "id": "clip-1"}],
            "library_id": library_id,
            "query": query,
        },
    )

    assert cli.main(["search", "--library", "beef-noodle", "--query", "厨师拉面"]) == 0

    assert json.loads(capsys.readouterr().out)["clips"][0]["id"] == "clip-1"
