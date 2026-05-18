import json


def test_insights_json_outputs_report(monkeypatch, capsys):
    from hermes_cli import main as main_mod

    closed = {"value": False}
    report = {
        "empty": False,
        "days": 7,
        "overview": {"estimated_cost": 0.04},
        "models": [],
    }

    class FakeSessionDB:
        def close(self):
            closed["value"] = True

    class FakeInsightsEngine:
        def __init__(self, db):
            self.db = db

        def generate(self, days=30, source=None):
            return {**report, "days": days, "source_filter": source}

        def format_terminal(self, generated_report):
            return "terminal report"

    monkeypatch.setattr("hermes_state.SessionDB", FakeSessionDB)
    monkeypatch.setattr("agent.insights.InsightsEngine", FakeInsightsEngine)
    monkeypatch.setattr("sys.argv", ["hermes", "insights", "--days", "7", "--source", "cli", "--json"])

    main_mod.main()

    data = json.loads(capsys.readouterr().out)
    assert data["days"] == 7
    assert data["source_filter"] == "cli"
    assert data["overview"]["estimated_cost"] == 0.04
    assert closed["value"] is True
