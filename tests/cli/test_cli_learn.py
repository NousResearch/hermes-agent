"""Regression tests for CLI /learn progress messaging."""

from tests.cli.test_cli_init import _make_cli


def test_process_command_learn_prints_progress_and_queues_prompt(capsys):
    cli = _make_cli()
    queued = []

    class _Queue:
        def put(self, value):
            queued.append(value)

    cli._pending_input = _Queue()

    cli.process_command("/learn the deploy workflow")

    out = capsys.readouterr().out
    assert "Learning a skill from what you described" in out
    assert "1/3 gather source material" in out
    assert "2/3 write and save the skill" in out
    assert "3/3 report the skill name and category" in out
    assert "new skill name" in out
    assert len(queued) == 1
    assert "the deploy workflow" in queued[0]
    assert "Keep progress visible" in queued[0]


def test_process_command_bare_learn_names_conversation_source(capsys):
    cli = _make_cli()
    queued = []

    class _Queue:
        def put(self, value):
            queued.append(value)

    cli._pending_input = _Queue()

    cli.process_command("/learn")

    out = capsys.readouterr().out
    assert "Learning a skill from this conversation" in out
    assert len(queued) == 1
    assert "workflow we just went through" in queued[0]
