"""Regression test: cronjob tool handler must pass attach_to_session.

Two bugs were fixed in ``tools/cronjob_tools.py``:

1. The handler lambda at the bottom of the file was missing
   ``attach_to_session=args.get("attach_to_session")``, so updating a
   job's ``attach_to_session`` via the tool (i.e. through the handler,
   not by calling ``cronjob()`` directly) silently dropped the parameter
   and returned "No updates provided."

2. ``_format_job`` did not include ``attach_to_session`` in its output
   dict, so the field was invisible in tool responses even when set.

The handler-level test (``test_handler_*``) is the critical one: it goes
through ``registry.dispatch()`` → handler lambda → ``cronjob()``, which
is the exact path the LLM takes.  The function-level tests
(``test_cronjob_*``) cover the direct ``cronjob()`` call and would pass
even with the handler bug present.
"""

import json

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)
    # Reload cronjob_tools so it re-registers with the (possibly reloaded)
    # registry singleton. Do NOT reload tools.registry itself — that creates
    # a fresh singleton and loses all registrations.
    import tools.cronjob_tools
    importlib.reload(tools.cronjob_tools)

    return home


# ---------------------------------------------------------------------------
# Handler-level tests — go through registry.dispatch() → handler lambda.
# These are the tests that actually catch the handler bug.
# ---------------------------------------------------------------------------


def test_handler_update_attach_to_session(hermes_env):
    """Updating attach_to_session via the tool handler must succeed.

    Before the fix, the handler lambda did not pass ``attach_to_session``
    to ``cronjob()``, so the update was silently dropped
    ("No updates provided.").
    """
    from tools.registry import registry

    create_result = json.loads(
        registry.dispatch("cronjob", {
            "action": "create",
            "schedule": "every 5m",
            "prompt": "daily briefing",
            "deliver": "local",
        })
    )
    assert create_result["success"] is True
    job_id = create_result["job_id"]

    # Update via the handler — this is the path that was broken.
    result = json.loads(
        registry.dispatch("cronjob", {
            "action": "update",
            "job_id": job_id,
            "attach_to_session": True,
        })
    )
    assert result["success"] is True, f"Update failed: {result.get('error')}"
    assert result["job"]["attach_to_session"] is True


def test_handler_create_attach_to_session(hermes_env):
    """Creating a job with attach_to_session via the handler must persist."""
    from tools.registry import registry

    created = json.loads(
        registry.dispatch("cronjob", {
            "action": "create",
            "schedule": "every 5m",
            "prompt": "conversational briefing",
            "deliver": "local",
            "attach_to_session": True,
        })
    )
    assert created["success"] is True
    assert created["job"]["attach_to_session"] is True


# ---------------------------------------------------------------------------
# Function-level tests — call cronjob() directly.
# These cover the cronjob() function itself and _format_job output.
# ---------------------------------------------------------------------------


def test_cronjob_tool_update_attach_to_session(hermes_env):
    """Direct cronjob() call: update must persist attach_to_session."""
    from tools.cronjob_tools import cronjob

    created = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            prompt="daily briefing",
            deliver="local",
        )
    )
    assert created["success"] is True
    job_id = created["job_id"]

    result = json.loads(
        cronjob(action="update", job_id=job_id, attach_to_session=True)
    )
    assert result["success"] is True
    assert result["job"]["attach_to_session"] is True
