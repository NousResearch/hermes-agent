"""Review-lane failures must name their cause, not surface as an ownership refusal.

``review_dispatch`` defaults true on the premise that the bundle carries ``sdlc-review``, but the
skill resolves from the profile's own skills dir. When it is absent (or operator-disabled) the worker
raises ``kanban_skills_unavailable`` **before its first model call** — and that failure used to reach
a human as two things that named nothing:

* the worker's own raise said only ``kanban_skills_unavailable``, with the skill left to be guessed
  from the review lane's source;
* the submitter saw ``Kanban execution unknown; retained by owner``, which reads as an ownership
  refusal, though the owner had simply never settled an admission for a worker that had already died.

Both now name the cause: the raise lists the skills it could not load, and an unsettled admission
caused by a worker that already wrote its exit reports that exit code.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_worker_names_the_skills_it_could_not_load(monkeypatch):
    from gateway import session_kanban

    monkeypatch.setattr(
        "agent.skill_commands.build_preloaded_skills_prompt",
        lambda identifiers, task_id=None, excluded_loaded_names=None: ("", [], ["sdlc-review"]),
    )
    agent = SimpleNamespace(session_id="t_contract")

    with pytest.raises(ValueError, match="kanban_skills_unavailable: sdlc-review"):
        session_kanban._run_task_turns(
            agent, {"text": "body"}, [], {"skills": ["sdlc-review"], "context": ""}
        )


def test_unsettled_admission_reports_the_worker_exit(monkeypatch):
    from hermes_cli import kanban_worker_client as client
    from hermes_cli.gateway_client import GatewayClientError

    monkeypatch.setattr("gateway.session_kanban.worker_result", lambda board_db, params: {"exit_code": 7})

    with pytest.raises(GatewayClientError, match="exited 7 before the owner settled the admission"):
        client._exit_code({"status": "unknown"}, {}, "board.db")
