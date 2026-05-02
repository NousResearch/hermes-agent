import sys


def test_camel_trace_subcommand_renders_json(monkeypatch, capsys):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(
        main_mod,
        "_load_camel_trace",
        lambda session_id=None: {
            "session_id": session_id or "session-1",
            "runtime_mode": "enforce",
            "summary": {"turn_count": 1, "policy_alert_count": 1, "response_alert_count": 0},
            "turns": [],
        },
    )
    monkeypatch.setattr(sys, "argv", ["hermes", "camel", "trace", "--format", "json", "--session", "abc123"])

    main_mod.main()

    output = capsys.readouterr().out
    assert '"session_id": "abc123"' in output
    assert '"runtime_mode": "enforce"' in output


def test_camel_benchmark_subcommand_renders_markdown(monkeypatch, capsys):
    import agent.camel_benchmark as bench_mod
    import hermes_cli.main as main_mod

    monkeypatch.setattr(bench_mod, "write_default_fixtures", lambda: {})
    monkeypatch.setattr(bench_mod, "run_response_comparison", lambda: [])
    monkeypatch.setattr(bench_mod, "run_policy_comparison", lambda: [])
    monkeypatch.setattr(bench_mod, "render_markdown", lambda response_rows, tool_rows: "# benchmark\n")
    monkeypatch.setattr(sys, "argv", ["hermes", "camel", "benchmark"])

    main_mod.main()

    assert capsys.readouterr().out == "# benchmark\n"


def test_camel_update_subcommand_delegates_to_cmd_update(monkeypatch):
    import hermes_cli.main as main_mod

    captured = {}

    def _fake_update(args):
        captured["camel_action"] = getattr(args, "camel_action", None)

    monkeypatch.setattr(main_mod, "cmd_update", _fake_update)
    monkeypatch.setattr(sys, "argv", ["hermes", "camel", "update"])

    main_mod.main()

    assert captured == {"camel_action": "update"}
