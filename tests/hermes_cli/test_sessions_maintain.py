import sys
import types
from types import SimpleNamespace

import hermes_state

from hermes_cli import sessions_cmd
from hermes_state_retention import RetentionCounts, RetentionReport, VacuumDecision


class _FakeDB:
    def __init__(self):
        self.calls = []
        self.closed = False

    def apply_retention_policy(self, policy, **kwargs):
        self.calls.append((policy, kwargs))
        report = RetentionReport(
            mode=policy.mode,
            dry_run=bool(kwargs.get("dry_run")),
            cutoff=1.0,
            totals=RetentionCounts(
                compacted_lineages=1,
                compacted_tool_results=2,
            ),
            by_source={
                "cron": RetentionCounts(
                    compacted_lineages=1,
                    compacted_tool_results=2,
                )
            },
            vacuum=VacuumDecision(reason="reclaimable bytes below threshold"),
        )
        return report

    def close(self):
        self.closed = True


def _install_config(monkeypatch, value):
    module = types.ModuleType("hermes_cli.config")
    module.load_config = lambda: value
    monkeypatch.setitem(sys.modules, "hermes_cli.config", module)


def test_maintain_dry_run_uses_configured_policy(monkeypatch, tmp_path, capsys):
    fake = _FakeDB()
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: fake)
    monkeypatch.setattr(sessions_cmd, "get_hermes_home", lambda: tmp_path)
    _install_config(
        monkeypatch,
        {"sessions": {"retention_mode": "layered", "retention_days": 90}},
    )
    args = SimpleNamespace(
        sessions_action="maintain",
        source="cron",
        dry_run=True,
        yes=False,
        no_vacuum=False,
    )

    assert sessions_cmd.cmd_sessions(args) == 0
    assert len(fake.calls) == 1
    policy, kwargs = fake.calls[0]
    assert policy.mode == "layered"
    assert kwargs["source"] == "cron"
    assert kwargs["dry_run"] is True
    assert fake.closed is True
    assert "Retention preview" in capsys.readouterr().out


def test_maintain_yes_applies_after_preview_without_vacuum(
    monkeypatch, tmp_path, capsys
):
    fake = _FakeDB()
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: fake)
    monkeypatch.setattr(sessions_cmd, "get_hermes_home", lambda: tmp_path)
    _install_config(monkeypatch, {"sessions": {"retention_mode": "layered"}})
    args = SimpleNamespace(
        sessions_action="maintain",
        source=None,
        dry_run=False,
        yes=True,
        no_vacuum=True,
    )

    assert sessions_cmd.cmd_sessions(args) == 0
    assert len(fake.calls) == 2
    assert fake.calls[0][1]["dry_run"] is True
    assert fake.calls[1][1].get("dry_run") is None
    assert fake.calls[1][1]["vacuum"] is False
    assert fake.closed is True
    assert "Retention complete" in capsys.readouterr().out
