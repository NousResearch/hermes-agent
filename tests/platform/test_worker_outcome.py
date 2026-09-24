"""A task whose run ended on a failed model call must say so on the board.

The live defect: Bedrock refused every call, the worker exited 0 in display mode, and the
dispatcher booked seven "protocol violation" runs across two tasks — advice that the work
had probably succeeded, and no trace of the provider's error. These run the *installed*
outcome plugin against a real board, the way a worker's exit would.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from nova.apply import apply_bundle
from nova.runtime.hermes.materialize import OUTCOME_PLUGIN_NAME
from nova.runtime.hermes.outcome import _needs_a_person
from nova.runtime.hermes.paths import HermesPaths

BEDROCK_DENIED = (
    "An error occurred (AccessDeniedException) when calling the Converse operation: Model access "
    "is denied due to IAM user or service role is not authorized to perform the required AWS "
    "Marketplace actions (NOT_AUTHORIZED)"
)


@pytest.mark.parametrize(
    "reason, status, retryable, expected",
    [
        ("auth", 403, True, True),           # Bedrock's NOT_AUTHORIZED: never fixes itself
        ("auth_permanent", 401, False, True),
        ("billing", 402, False, True),
        ("model_not_found", 404, False, True),
        ("content_policy_blocked", 400, False, True),
        ("rate_limit", 429, True, False),
        ("overloaded", 529, True, False),
        ("server_error", 500, True, False),
        ("timeout", None, True, False),
        ("unknown", None, None, False),       # no verdict: give it the retry budget
        ("unknown", None, False, True),       # the runtime says it will not work again
    ],
)
def test_which_failures_stop_for_a_person(reason, status, retryable, expected):
    assert _needs_a_person(reason, status, retryable) is expected


@pytest.fixture
def board(bundle, runtime, audit, home, monkeypatch):
    kb = pytest.importorskip("hermes_cli.kanban_db", reason="writes through the runtime's API")
    from hermes_cli import kanban_db_connect as kbc

    monkeypatch.setenv("HERMES_HOME", str(home))
    apply_bundle(bundle, runtime, audit=audit)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path()))
    kbc.init_db()
    return kb, kbc


def installed(home, agent="operations"):
    plugin_dir = HermesPaths(home=home).outcome_plugin_dir(agent)
    name = f"nova_outcome_{len(sys.modules)}"
    spec = importlib.util.spec_from_file_location(
        name, plugin_dir / "__init__.py", submodule_search_locations=[str(plugin_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def running_task(board, monkeypatch):
    kb, kbc = board
    with kbc.connect_closing() as c:
        task_id = kb.create_task(c, title="audit", assignee="operations", created_by="nova-supervisor", tenant="acme")
        claimed = kb.claim_task(c, task_id)  # what the dispatcher does before it spawns
        assert claimed is not None and claimed.status == "running"
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    return task_id


def events(board, task_id, kind):
    _, kbc = board
    with kbc.connect_closing() as c:
        return [json.loads(r[0] or "{}") for r in c.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind=?", (task_id, kind))]


def test_every_agent_gets_the_plugin_enabled(bundle, runtime, audit, home):
    import yaml

    apply_bundle(bundle, runtime, audit=audit)
    for agent in ("customer-support", "operations"):
        profile = HermesPaths(home=home).profile_dir(agent)
        assert (profile / "plugins" / OUTCOME_PLUGIN_NAME / "_model_errors.py").is_file()
        config = yaml.safe_load((profile / "config.yaml").read_text())
        assert OUTCOME_PLUGIN_NAME in config["plugins"]["enabled"]


def test_a_refused_model_blocks_the_task_with_the_providers_words(board, home, monkeypatch):
    task_id = running_task(board, monkeypatch)
    plugin = installed(home)
    plugin.on_api_request_error(error={"type": "AccessDeniedException", "message": BEDROCK_DENIED},
                                reason="auth", status_code=403, retryable=True,
                                provider="bedrock", model="eu.anthropic.claude-sonnet-4-6")
    assert plugin.settle() == "blocked"
    kb, kbc = board
    with kbc.connect_closing() as c:
        assert kb.get_task(c, task_id).status == "blocked"
    (blocked,) = events(board, task_id, "blocked")
    assert "NOT_AUTHORIZED" in blocked["reason"] and "HTTP 403" in blocked["reason"]
    assert blocked["reason"].count("protocol violation") == 0


def test_a_throttled_model_is_left_to_the_retry_with_the_cause_on_the_card(board, home, monkeypatch):
    task_id = running_task(board, monkeypatch)
    plugin = installed(home)
    plugin.on_api_request_error(error={"message": "ThrottlingException: Too many requests"},
                                reason="rate_limit", status_code=429, retryable=True)
    assert plugin.settle() == "commented"
    kb, kbc = board
    with kbc.connect_closing() as c:
        assert kb.get_task(c, task_id).status == "running", "the dispatcher's retry decides"
        comments = [r[0] for r in c.execute("SELECT body FROM task_comments WHERE task_id=?", (task_id,))]
    assert any("Too many requests" in body for body in comments)


def test_a_later_successful_call_clears_the_failure(board, home, monkeypatch):
    """A fallback model that answered means the run did not fail on the model."""
    running_task(board, monkeypatch)
    plugin = installed(home)
    plugin.on_api_request_error(error={"message": BEDROCK_DENIED}, reason="auth", status_code=403)
    plugin.on_post_api_request()
    assert plugin.settle() is None


def test_a_task_the_agent_already_closed_is_not_touched(board, home, monkeypatch):
    task_id = running_task(board, monkeypatch)
    kb, kbc = board
    with kbc.connect_closing() as c:
        kb.complete_task(c, task_id, summary="done")
    plugin = installed(home)
    plugin.on_api_request_error(error={"message": BEDROCK_DENIED}, reason="auth", status_code=403)
    assert plugin.settle() is None


def test_outside_a_task_nothing_is_written(board, home, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    plugin = installed(home)
    plugin.on_api_request_error(error={"message": BEDROCK_DENIED}, reason="auth", status_code=403)
    assert plugin.settle() is None
