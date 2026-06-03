import argparse

from hermes_cli.eval_lab import cmd_eval_lab_run


def test_eval_lab_command_registry_entry_exists():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("eval-lab")

    assert cmd.name == "eval-lab"
    assert "run" in cmd.subcommands
    assert cmd.cli_only is True


def test_cmd_eval_lab_run_writes_local_artifacts(tmp_path, capsys):
    scenario_file = tmp_path / "scenarios.yaml"
    scenario_file.write_text(
        """
scenarios:
  - id: cli_probe
    title: CLI probe
    prompt: Say hello.
    tags: [smoke]
    expected_artifacts: []
    blocked_actions: []
    success_criteria:
      - Echo
""".strip(),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        eval_lab_command="run",
        scenarios=str(scenario_file),
        run_id="cli-run",
        attempts=1,
        output_dir=str(tmp_path / "runs"),
    )

    exit_code = cmd_eval_lab_run(args)

    captured = capsys.readouterr()
    run_dir = tmp_path / "runs" / "cli-run"
    assert exit_code == 0
    assert "cli-run" in captured.out
    assert (run_dir / "trajectory_groups.jsonl").exists()
    assert (run_dir / "scores.jsonl").exists()
    assert (run_dir / "report.md").exists()
