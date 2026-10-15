"""Webhook cron references belong to the route's profile (#139332)."""

import json
from argparse import Namespace

import pytest

from hermes_cli import webhook
from hermes_constants import get_hermes_home


@pytest.fixture
def homes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(webhook, "_is_webhook_enabled", lambda: True)
    monkeypatch.setattr(webhook, "_get_webhook_base_url", lambda: "http://localhost:8644")
    coder = tmp_path / "profiles" / "coder"
    coder.mkdir(parents=True)
    (coder / "config.yaml").write_text("{}\n", encoding="utf-8")
    for home, job_id in ((tmp_path, "default-job"), (coder, "coder-job")):
        cron = home / "cron"
        cron.mkdir()
        (cron / "jobs.json").write_text(json.dumps({"jobs": [
            {"id": job_id, "name": "daily", "enabled": True}
        ]}), encoding="utf-8")
    return tmp_path, coder


def subscribe(**kwargs):
    args = dict(webhook_action="subscribe", name="route", secret="fixture-secret",
                prompt="", events="", description="", skills="", deliver="log",
                deliver_chat_id="", route_profile=None, cron_job="daily")
    args.update(kwargs)
    webhook.webhook_command(Namespace(**args))


@pytest.mark.parametrize("reference", ["coder-job", "daily"])
def test_subscribe_resolves_cron_in_route_profile(homes, reference):
    root, coder = homes
    subscribe(route_profile="coder", cron_job=reference)
    stored = webhook._load_subscriptions()
    assert stored.get("route", {}).get("cron_job") == "coder-job"
    assert stored["route"]["profile"] == "coder"
    assert get_hermes_home() == root
    assert not (coder / "webhook_subscriptions.json").exists()


def test_update_keeps_existing_profile_for_cron_lookup(homes):
    subscribe(route_profile="coder", cron_job="")
    subscribe()
    assert webhook._load_subscriptions()["route"]["cron_job"] == "coder-job"


def test_default_route_still_resolves_default_job(homes):
    subscribe()
    assert webhook._load_subscriptions()["route"]["cron_job"] == "default-job"


@pytest.mark.parametrize("failure", ["missing", "ambiguous", "corrupt"])
def test_failed_target_lookup_preserves_route_and_restores_store(homes, failure):
    from cron.jobs import resolve_job_ref

    root, coder = homes
    subscribe(route_profile="coder", cron_job="")
    before = (root / "webhook_subscriptions.json").read_bytes()
    store = coder / "cron" / "jobs.json"
    if failure == "missing":
        store.write_text('{"jobs": []}', encoding="utf-8")
    elif failure == "ambiguous":
        store.write_text(json.dumps({"jobs": [
            {"id": "one", "name": "daily"}, {"id": "two", "name": "daily"}
        ]}), encoding="utf-8")
    else:
        store.write_text("not json", encoding="utf-8")
    if failure == "corrupt":
        with pytest.raises(RuntimeError, match="corrupted"):
            subscribe()
    else:
        subscribe()
    assert (root / "webhook_subscriptions.json").read_bytes() == before
    assert resolve_job_ref("daily")["id"] == "default-job"
    assert get_hermes_home() == root
    assert not (coder / "webhook_subscriptions.json").exists()


def test_bare_route_preserves_custom_caller_home(homes, monkeypatch):
    root, _ = homes
    custom = root / "standalone"
    (custom / "cron").mkdir(parents=True)
    (custom / "cron" / "jobs.json").write_text(json.dumps({"jobs": [
        {"id": "custom-job", "name": "daily"}
    ]}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(custom))
    subscribe()
    assert webhook._load_subscriptions()["route"]["cron_job"] == "custom-job"
    assert not (root / "webhook_subscriptions.json").exists()
