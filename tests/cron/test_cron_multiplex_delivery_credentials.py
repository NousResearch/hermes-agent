"""Cron delivery keeps the firing profile's credentials for the full job lifecycle.

A multiplex ticker may run under profile A while ticking profile B with no live
adapters. Both successful output and escaped-failure alerts must reach the
standalone Discord sender with B's credential, then restore the caller's empty
secret scope.
"""

import pytest

import cron.scheduler as scheduler


@pytest.fixture()
def secondary_discord_delivery(tmp_path, monkeypatch):
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import send_message_tool

    primary_home = tmp_path / "primary"
    secondary_home = tmp_path / "profiles" / "secondary"
    primary_home.mkdir()
    secondary_home.mkdir(parents=True)
    (secondary_home / ".env").write_text(
        "DISCORD_BOT_TOKEN=profile-b-credential-value\n",
        encoding="utf-8",
    )
    (secondary_home / "config.yaml").write_text(
        "platforms:\n  discord:\n    enabled: true\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(primary_home))
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "profile-a-credential-value")
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda _job: False)
    monkeypatch.setattr(
        scheduler,
        "create_execution",
        lambda job_id, **_kwargs: {"id": f"exec-{job_id}"},
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda _execution_id: {})
    monkeypatch.setattr(
        scheduler,
        "save_job_output",
        lambda job_id, _output: str(tmp_path / f"{job_id}.md"),
    )
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(scheduler, "finish_execution", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scheduler, "load_config", lambda: {"cron": {"wrap_response": False}}
    )
    monkeypatch.setattr(
        scheduler,
        "_upsert_incident_for_failure",
        lambda *_args, **_kwargs: (False, "incident-test"),
    )
    monkeypatch.setattr(scheduler, "_mark_incident_alerted", lambda _incident_id: None)

    sends = []

    async def fake_standalone_sender(pconfig, chat_id, message, **_kwargs):
        sends.append({
            "credential": pconfig.token,
            "chat_id": chat_id,
            "message": message,
        })
        return {"success": True, "message_id": "message-test"}

    original_sender_lookup = send_message_tool._plugin_standalone_sender

    def standalone_sender_lookup(platform_name, **kwargs):
        if platform_name == "discord":
            return fake_standalone_sender, None
        return original_sender_lookup(platform_name, **kwargs)

    monkeypatch.setattr(
        send_message_tool,
        "_plugin_standalone_sender",
        standalone_sender_lookup,
    )

    previous_multiplex = secret_scope.is_multiplex_active()
    empty_scope_token = secret_scope.set_secret_scope(None)
    home_token = set_hermes_home_override(str(secondary_home))
    secret_scope.set_multiplex_active(True)
    try:
        yield sends, secret_scope
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        reset_hermes_home_override(home_token)
        secret_scope.reset_secret_scope(empty_scope_token)


def test_secondary_normal_delivery_retains_profile_scope(
    secondary_discord_delivery, monkeypatch
):
    sends, secret_scope = secondary_discord_delivery
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_args, **_kwargs: (True, "raw output", "normal result", None),
    )

    processed = scheduler.run_one_job(
        {"id": "normal", "name": "normal", "deliver": "discord:channel-b"},
        adapters=None,
        loop=None,
    )

    assert processed is True
    assert [send["credential"] for send in sends] == ["profile-b-credential-value"]
    assert [send["chat_id"] for send in sends] == ["channel-b"]
    assert "normal result" in sends[0]["message"]
    assert secret_scope.current_secret_scope() is None


def test_secondary_failure_alert_retains_profile_scope(
    secondary_discord_delivery, monkeypatch
):
    sends, secret_scope = secondary_discord_delivery

    def raise_during_run(*_args, **_kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(scheduler, "run_job", raise_during_run)

    processed = scheduler.run_one_job(
        {"id": "failure", "name": "failure", "deliver": "discord:channel-b"},
        adapters=None,
        loop=None,
    )

    assert processed is False
    assert [send["credential"] for send in sends] == ["profile-b-credential-value"]
    assert [send["chat_id"] for send in sends] == ["channel-b"]
    assert "failed" in sends[0]["message"].lower()
    assert secret_scope.current_secret_scope() is None
