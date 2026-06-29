from gateway.progress_summary import ToolProgressSummary


def test_summary_hides_raw_hermes_command_but_shows_profile_and_kanban_task():
    summary = ToolProgressSummary()

    text = summary.update(
        "tool.started",
        "terminal",
        args={
            "command": (
                "HERMES_KANBAN_BOARD=mes hermes -p mes-release-run "
                "--accept-hooks --skills kanban-agent-workflows "
                "--workdir /a0/usr/projects/MES chat -q \"work kanban task t_62b0683f\""
            )
        },
    )

    assert text is not None
    assert "Profil mes-release-run" in text
    assert "t_62b0683f" in text
    assert "HERMES_KANBAN_BOARD" not in text
    assert "--accept-hooks" not in text
    assert "chat -q" not in text
    assert "outils 0/1 terminés" in text


def test_summary_updates_tool_completion_counters_and_errors():
    summary = ToolProgressSummary()

    summary.update("tool.started", "read_file", args={"path": "AGENTS.md"})
    text = summary.update("tool.completed", "read_file", duration=1.25, is_error=False)

    assert text is not None
    assert "read_file terminé en 1.2s" in text
    assert "outils 1/1 terminés" in text

    summary.update("tool.started", "patch", args={"path": "gateway/run.py"})
    text = summary.update("tool.completed", "patch", duration=0.5, is_error=True)

    assert text is not None
    assert "erreurs 1" in text
    assert "outils 2/2 terminés" in text


def test_summary_renders_process_wait_as_process_tracking():
    summary = ToolProgressSummary()

    text = summary.update("tool.started", "process", preview="wait proc_a2671d234bd 240s")

    assert text is not None
    assert "Étape: Suivi processus" in text
    assert "Attente processus proc_a2671d234bd (240s)" in text


def test_summary_tracks_subagents_without_mirroring_child_text():
    summary = ToolProgressSummary()

    text = summary.update(
        "subagent.spawn_requested",
        preview="Audit du module Telegram",
        task_index=0,
        model="openai/gpt-test",
        goal="Audit du module Telegram et proposition de correctif",
    )
    assert text is not None
    assert "Agents: 1 actif(s), 0 terminé(s)" in text
    assert "#1 openai/gpt-test" in text

    text = summary.update(
        "subagent.tool",
        "search_files",
        preview="telegram",
        task_index=0,
        model="openai/gpt-test",
    )
    assert text is not None
    assert "outils agents 1" in text
    assert "search_files" in text

    text = summary.update(
        "subagent.complete",
        preview="Correctif prêt",
        task_index=0,
        model="openai/gpt-test",
    )
    assert text is not None
    assert "Agents: 0 actif(s), 1 terminé(s)" in text


def test_summary_tracks_moa_reference_and_aggregation():
    summary = ToolProgressSummary()

    text = summary.update(
        "moa.reference",
        "ref-model-a",
        "réponse longue non recopiée ici",
        moa_index=0,
        moa_count=2,
    )
    assert text is not None
    assert "Étape: Consultation MoA" in text
    assert "1/2" in text

    text = summary.update("moa.aggregating", "aggregator", moa_ref_count=2)
    assert text is not None
    assert "Étape: Synthèse MoA" in text
    assert "Agrégation de 2 réponse(s)" in text
