class _FakeAgent:
    def __init__(self):
        self._nudge_disabled = False


def test_skills_nudge_off_sets_flag():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = _FakeAgent()

    cli._handle_skills_command("/skills nudge off")

    assert cli.agent._nudge_disabled is True


def test_skills_nudge_on_clears_flag():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = _FakeAgent()
    cli.agent._nudge_disabled = True

    cli._handle_skills_command("/skills nudge on")

    assert cli.agent._nudge_disabled is False


def test_other_skills_subcommand_delegates_to_hub(monkeypatch):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = _FakeAgent()
    called = {}

    def fake_handle(cmd, console):
        called["cmd"] = cmd

    monkeypatch.setattr("hermes_cli.skills_hub.handle_skills_slash", fake_handle)
    cli._handle_skills_command("/skills list")

    assert called["cmd"] == "/skills list"
