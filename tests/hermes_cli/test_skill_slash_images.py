"""Images attached on the same Enter as a slash command must never vanish with it."""
import queue

import pytest

from cli import HermesCLI


def _cli(process_command):
    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    cli._pending_agent_seed = None
    cli._pending_resume_sessions = []
    cli._attached_images = []
    cli._should_exit = False
    cli._status_bar_suppressed_after_resize = False
    cli._agent_running = False
    cli._interactive_turn = False
    cli._app = type("App", (), {"invalidate": lambda self: None, "is_running": False, "exit": lambda self: None})()
    cli.process_command = lambda text: process_command(cli, text) or True
    cli._typed_voice_stop = lambda text: False
    cli.handle_bang_shell = lambda text: False
    cli._turn_summary_begin = lambda: None
    cli._tui_after_turn = lambda: None
    cli._print_user_message_preview = lambda text: None
    cli.seen = []
    cli.chat = lambda text, images=None, voice_input=False: cli.seen.append((text, images))
    return cli


@pytest.mark.parametrize("queue_turn", [
    lambda cli: cli._queue_skill_message("QUEUED TURN"),          # /<skill> ...
    lambda cli: cli._kick_goal("QUEUED TURN"),                    # /goal <objective>
    lambda cli: cli._queue_prompt_turn("QUEUED TURN", "/plan"),   # /plan, /learn, /init
    lambda cli: setattr(cli, "_pending_agent_seed", "QUEUED TURN"),  # seed-returning commands
], ids=["skill", "goal", "plan", "seed"])
def test_turn_queuing_slash_command_carries_the_attached_images(queue_turn):
    cli = _cli(lambda cli, text: queue_turn(cli))

    cli._tui_process_one_input(("/cmd look at this screenshot", ["shot.png"]))
    if not cli._pending_input.empty():
        cli._tui_process_one_input(cli._pending_input.get_nowait())

    assert cli.seen == [("QUEUED TURN", ["shot.png"])]
    assert cli._pending_input.empty() and cli._attached_images == []


def test_non_queuing_slash_command_puts_the_images_back_on_the_composer():
    cli = _cli(lambda cli, text: None)  # e.g. /help, /model: runs, queues no turn

    cli._tui_process_one_input(("/help", ["shot.png"]))

    assert cli.seen == [] and cli._pending_input.empty()
    assert cli._attached_images == ["shot.png"]
