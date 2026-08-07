import argparse

from hermes_cli.smoke import render_smoke_report, run_smoke


def test_run_smoke_default_does_not_write_summary_or_chat(monkeypatch, tmp_path):
    calls = []

    def fake_smoke_command(name, cmd, expected_substring, artifact_dir, timeout=120):
        calls.append((name, cmd, expected_substring, artifact_dir))
        from hermes_cli.smoke import SmokeResult

        return SmokeResult(name, True, "rc=0", 0.01, None)

    monkeypatch.setattr("hermes_cli.smoke.smoke_command", fake_smoke_command)

    data = run_smoke(profiles=["default"])

    assert data["ok"] is True
    assert data["artifact_dir"] is None
    assert not (tmp_path / "summary.json").exists()
    assert all(call[3] is None for call in calls)
    names = [item["name"] for item in data["results"]]
    assert "version" in names
    assert "doctor" in names
    assert "context-audit" in names
    assert not any(name.startswith("profile:") for name in names)
    assert "not written" in render_smoke_report(data)


def test_run_smoke_writes_artifacts_when_requested(monkeypatch, tmp_path):
    def fake_smoke_command(name, cmd, expected_substring, artifact_dir, timeout=120):
        from hermes_cli.smoke import SmokeResult

        assert artifact_dir == tmp_path
        return SmokeResult(name, True, "rc=0", 0.01, str(artifact_dir / f"{name}.stdout"))

    monkeypatch.setattr("hermes_cli.smoke.smoke_command", fake_smoke_command)

    data = run_smoke(profiles=["default"], artifact_dir=tmp_path)

    assert data["artifact_dir"] == str(tmp_path)
    assert (tmp_path / "summary.json").exists()


def test_run_smoke_chat_is_opt_in(monkeypatch, tmp_path):
    profile_calls = []

    def fake_smoke_command(name, cmd, expected_substring, artifact_dir, timeout=120):
        from hermes_cli.smoke import SmokeResult

        return SmokeResult(name, True, "rc=0", 0.01, None)

    def fake_smoke_profile(profile, artifact_dir, *, cli=None, timeout=180):
        profile_calls.append((profile, artifact_dir, cli))
        from hermes_cli.smoke import SmokeResult

        return SmokeResult(f"profile:{profile}", True, "rc=0", 0.01, None)

    monkeypatch.setattr("hermes_cli.smoke.smoke_command", fake_smoke_command)
    monkeypatch.setattr("hermes_cli.smoke.smoke_profile", fake_smoke_profile)

    data = run_smoke(profiles=["default"], chat=True, cli="/bin/hermes-test")

    assert data["ok"] is True
    assert profile_calls == [("default", None, "/bin/hermes-test")]


def test_smoke_uses_configurable_cli_argument():
    from hermes_cli.smoke import _hermes_cmd

    assert _hermes_cmd("/tmp/fake-hermes") == ["/tmp/fake-hermes"]


def test_smoke_parser_has_no_placeholder_flags():
    from hermes_cli.subcommands.smoke import build_smoke_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_smoke_parser(subparsers, cmd_smoke=lambda args: None)

    help_text = parser.format_help()
    assert "--browser" not in help_text
    assert "--delegation" not in help_text
    args = parser.parse_args(["smoke", "--chat", "--profiles", "default", "--cli", "/bin/hermes"])
    assert args.chat is True
    assert args.profiles == "default"
    assert args.cli == "/bin/hermes"
