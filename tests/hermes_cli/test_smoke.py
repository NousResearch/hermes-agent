import argparse

from hermes_cli.smoke import render_smoke_report, run_smoke, smoke_profile


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


def test_smoke_profile_accepts_sentinel_in_non_exact_output(monkeypatch):
    monkeypatch.setattr("hermes_cli.smoke._profile_exists", lambda profile: True)

    def fake_run(cmd, timeout=120, env=None):
        return 0, "Here is the result: SMOKE_default_OK\n", "", 0.01

    monkeypatch.setattr("hermes_cli.smoke._run", fake_run)

    result = smoke_profile("default", None, cli="/bin/hermes-test")

    assert result.ok is True
    assert "expected token present in non-exact output" in result.detail


def test_smoke_profile_skips_missing_named_profile(monkeypatch):
    called = False

    def fake_run(cmd, timeout=120, env=None):
        nonlocal called
        called = True
        return 0, "SMOKE_lab_OK", "", 0.01

    monkeypatch.setattr("hermes_cli.smoke._profile_exists", lambda profile: False)
    monkeypatch.setattr("hermes_cli.smoke._run", fake_run)

    result = smoke_profile("lab", None, cli="/bin/hermes-test")

    assert result.ok is True
    assert "skipped; profile not found" in result.detail
    assert called is False


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
