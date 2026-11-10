"""External cron workers must register the owning profile's config-declared hooks (#131764).

``python -m cron.scheduler --external-worker-file`` starts with the builtin registry alone and
used to never call ``shell_hooks.register_from_config()`` / ``outbound_webhooks.register_from_
config()``, so ``hooks:`` blocks in the owning profile's config.yaml silently stopped firing for
cron sessions the moment fires were handed to the worker instead of running in-process in the
gateway — plugin hooks kept working (``discover_plugins()`` ran) and nothing was logged. Both
registrations must run under the payload's home override so a multiplexed worker registers the
OWNING profile's hooks, and consent must resolve non-interactively (``hooks_auto_accept`` /
allowlist / env — this process has no TTY).
"""
from __future__ import annotations

import json

import pytest

HOOK_CMD = "echo cron-hook-fired"
OUTBOUND_URL = "https://hooks.example.test/cron"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A launch home with no hooks config and a profile home declaring both hook kinds."""
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text(
        "hooks:\n"
        "  on_session_start:\n"
        f"    - command: \"{HOOK_CMD}\"\n"
        "  outbound:\n"
        f"    - url: {OUTBOUND_URL}\n"
        "      events: [on_session_end]\n"
        # Resolved through get_secret: raises outside a secret scope while multiplexing.
        "      secret_env: CRON_HOOK_SECRET\n"
        # Non-interactive consent: the worker has no TTY, and the pair is not allowlisted.
        "hooks_auto_accept: true\n",
        encoding="utf-8",
    )
    (profile / ".env").write_text("CRON_HOOK_SECRET=profile-secret\n", encoding="utf-8")
    yield launch, profile
    from agent import outbound_webhooks
    from agent import shell_hooks
    from hermes_cli.plugins import _reset_plugin_managers_for_tests

    shell_hooks.reset_for_tests()
    outbound_webhooks.reset_for_tests()
    _reset_plugin_managers_for_tests()


def test_worker_registers_owning_profile_config_hooks(homes, tmp_path, monkeypatch):
    _launch, profile = homes
    from agent import outbound_webhooks
    from agent import shell_hooks
    from cron import scheduler
    from hermes_constants import hermes_home_key

    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({"job": {"id": "job-1", "execution_id": "exec-1"},
                    "profile_home": str(profile), "multiplex_active": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cron.executions.adopt_claimed_execution",
        lambda execution_id: {"id": execution_id, "status": "running"},
    )
    monkeypatch.setattr(scheduler, "run_one_job", lambda *a, **k: True)

    assert scheduler._run_external_worker_payload(payload, tmp_path / "exec-1.ready") is True
    # Registered under the OWNING profile's home key, not the launch home's.
    profile_key = hermes_home_key(profile)
    assert any(home == profile_key and event == "on_session_start" and command == HOOK_CMD
               for home, event, _matcher, command in shell_hooks._registered)
    assert any(home == profile_key and event == "on_session_end" and url == OUTBOUND_URL
               for home, event, url in outbound_webhooks._registered)
