from __future__ import annotations


def test_ard_regress_cli_runs_intent_regression(monkeypatch, capsys):
    calls = []

    def fake_main(argv):
        calls.append(argv)
        print('{"ok": true, "summary": {"total": 1, "passed": 1, "failed": 0}}')
        return 0

    monkeypatch.setattr("scripts.ard_intent_regression.main", fake_main)
    from cli import HermesCLI
    cli = object.__new__(HermesCLI)
    cli._handle_ard_command("/ard regress --json --output /tmp/report.json --query youtube transcript")
    assert calls == [["--json", "--output", "/tmp/report.json", "--query", "youtube transcript"]]
    assert '"ok": true' in capsys.readouterr().out


def test_ard_compare_search_cli_runs_spike(monkeypatch, capsys):
    calls = []

    def fake_main(argv):
        calls.append(argv)
        print('{"schema": "hermes.ard.skill-search-spike.v1", "external_available": false}')
        return 0

    monkeypatch.setattr("scripts.ard_skill_search_spike.main", fake_main)
    from cli import HermesCLI
    cli = object.__new__(HermesCLI)
    cli._handle_ard_command("/ard compare-search --json --output /tmp/compare.json --query browser qa")
    assert calls == [["--json", "--output", "/tmp/compare.json", "--query", "browser qa"]]
    assert "skill-search-spike" in capsys.readouterr().out
