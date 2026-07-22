import subprocess


def test_build_goal_workspace_title_compacts_goal_text():
    from hermes_cli.cmux_integration import build_goal_workspace_title

    title = build_goal_workspace_title(
        "  /goal  Ship cmux workspace auto-title support\n\n"
        "- update classic CLI\n"
        "- update TUI\n",
        max_chars=48,
    )

    assert title == "Goal: Ship cmux workspace auto-title support"
    assert "\n" not in title
    assert len(title) <= 48


def test_build_goal_workspace_title_truncates_at_character_boundary():
    from hermes_cli.cmux_integration import build_goal_workspace_title

    title = build_goal_workspace_title(
        "cmux の workspace タイトルを /goal の内容に合わせて自動更新する",
        max_chars=24,
    )

    assert title.startswith("Goal: cmux の workspace")
    assert title.endswith("…")
    assert len(title) <= 24


def test_rename_cmux_workspace_for_goal_runs_cmux_when_inside_cmux():
    from hermes_cli.cmux_integration import rename_cmux_workspace_for_goal

    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    title = rename_cmux_workspace_for_goal(
        "Improve the /goal title handoff",
        config={"cmux": {"auto_rename_workspace_on_goal": True}},
        env={"CMUX_WORKSPACE_ID": "workspace:12"},
        runner=runner,
    )

    assert title == "Goal: Improve the /goal title handoff"
    assert calls == [
        (
            [
                "cmux",
                "rename-workspace",
                "--workspace",
                "workspace:12",
                "Goal: Improve the /goal title handoff",
            ],
            {"check": False, "capture_output": True, "text": True, "timeout": 2.0},
        )
    ]


def test_rename_cmux_workspace_for_goal_skips_outside_cmux():
    from hermes_cli.cmux_integration import rename_cmux_workspace_for_goal

    calls = []
    title = rename_cmux_workspace_for_goal(
        "No cmux env",
        config={"cmux": {"auto_rename_workspace_on_goal": True}},
        env={},
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert title is None
    assert calls == []


def test_rename_cmux_workspace_for_goal_respects_disabled_config():
    from hermes_cli.cmux_integration import rename_cmux_workspace_for_goal

    calls = []
    title = rename_cmux_workspace_for_goal(
        "Disabled",
        config={"cmux": {"auto_rename_workspace_on_goal": False}},
        env={"CMUX_WORKSPACE_ID": "workspace:12"},
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert title is None
    assert calls == []
