"""``hermes -z`` keeps the terminal quiet without silencing the log files (#134971).

``logging.disable(CRITICAL)`` is a manager-wide threshold checked before any handler, so it
also dropped every record the queued file handlers needed — one-shot runs never wrote
agent.log, including the per-call ``API call #N: ... upstream=<provider>`` line the gateway
path always logs. Quiet is now per-handler: terminal-bound handlers are raised past CRITICAL
for the run, the queue handler feeding agent.log is untouched, and levels are restored after."""

import logging
from unittest import mock

import pytest

import hermes_cli.oneshot as oneshot


class _RecordingHandler(logging.Handler):
    def __init__(self, level: int = logging.DEBUG):
        super().__init__(level=level)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def root_pipeline():
    """Root logger carrying a file-pipeline queue handler plus a terminal-bound handler.

    Mirrors a live process: ``setup_logging()`` attaches RotatingFileHandlers through a root
    queue handler marked ``_hermes_queue``, while the ``--verbose`` console handler holds the
    real stderr stream. Everything is restored afterwards so other tests are unaffected."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    queue_like = _RecordingHandler(level=logging.INFO)
    queue_like._hermes_queue = True  # type: ignore[attr-defined]  # the agent.log pipeline entry
    terminal = _RecordingHandler(
        level=logging.DEBUG
    )  # e.g. the --verbose stderr handler
    root.handlers = [queue_like, terminal]
    root.setLevel(logging.INFO)
    yield queue_like, terminal
    root.handlers = saved_handlers
    root.setLevel(saved_level)


def test_run_keeps_file_pipeline_records_live(monkeypatch, tmp_path, root_pipeline):
    queue_like, terminal = root_pipeline

    def fake_run_agent(*args, **kwargs):
        logging.getLogger("agent.conversation_loop").info(
            "API call #1: model=m provider=p in=1 out=2 total=3"
        )
        return "ok", {"final_response": "ok", "completed": True}

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with mock.patch.object(oneshot, "_run_agent", side_effect=fake_run_agent):
        code = oneshot.run_oneshot("q")

    assert code == 0
    assert any("API call #1" in r.getMessage() for r in queue_like.records)
    assert not terminal.records  # the terminal stayed quiet for the whole run


def test_terminal_handler_levels_restored_after_run(
    monkeypatch, tmp_path, root_pipeline
):
    _, terminal = root_pipeline

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with mock.patch.object(
        oneshot, "_run_agent", return_value=("ok", {"completed": True})
    ):
        oneshot.run_oneshot("q")

    assert terminal.level == logging.DEBUG


def test_levels_restored_even_when_agent_raises(monkeypatch, tmp_path, root_pipeline):
    _, terminal = root_pipeline

    def boom(*args, **kwargs):
        raise RuntimeError("provider exploded")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with mock.patch.object(oneshot, "_run_agent", side_effect=boom):
        code = oneshot.run_oneshot("q")

    assert code == 1
    assert terminal.level == logging.DEBUG
