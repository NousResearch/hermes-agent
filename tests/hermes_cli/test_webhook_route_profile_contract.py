"""Reported routed webhook cron lookup uses the target's actual store (#139332)."""

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.mark.parametrize("reference", ["coder-job", "same-name"])
def test_named_route_looks_up_its_job_without_moving_subscription_home(
    tmp_path, monkeypatch, reference
):
    from hermes_cli import webhook
    from cron.jobs import resolve_job_ref
    from hermes_constants import get_hermes_home

    root = tmp_path / "install"
    coder = root / "profiles" / "coder"
    for home, job_id in ((root, "launch-job"), (coder, "coder-job")):
        (home / "cron").mkdir(parents=True)
        (home / "config.yaml").write_text(
            "platforms:\n  webhook:\n    enabled: true\n", encoding="utf-8"
        )
        (home / "cron" / "jobs.json").write_text(
            json.dumps({"jobs": [{"id": job_id, "name": "same-name"}]}),
            encoding="utf-8",
        )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    args = Namespace(
        webhook_action="subscribe", name="notify", route_profile="coder",
        cron_job=reference, secret="synthetic-secret", prompt="", events="",
        description="", skills="", deliver="log", deliver_chat_id="",
    )
    webhook.webhook_command(args)
    assert webhook._load_subscriptions().get("notify", {}).get("cron_job") == "coder-job"
    assert get_hermes_home() == root
    assert resolve_job_ref("same-name")["id"] == "launch-job"
    assert not (coder / "webhook_subscriptions.json").exists()
    args.route_profile = None
    args.cron_job = "same-name"
    webhook.webhook_command(args)
    assert webhook._load_subscriptions()["notify"]["cron_job"] == "coder-job"


@pytest.mark.parametrize(("arguments", "expected"), [
    (["--route-profile", "coder", "--cron-job", "missing-job"], "no cron job matches"),
    (["--route-profile", "coder", "--cron-job", "same"], "ambiguous"),
    (["--route-profile", "absent"], "does not exist"),
    (["--route-profile", "../outside"], "Error:"),
    (["--route-profile", "default", "--secret", "fixture-secret"], "requires a new HMAC secret"),
    (["--route-profile", "coder", "--cron-job", "missing-job", "--deliver-only", "--deliver", "telegram"],
     "mutually exclusive"),
], ids=["missing", "ambiguous", "missing_profile", "invalid_profile", "same_secret", "deliver_cron"])
def test_cli_refused_target_job_exits_nonzero_without_replacing_route(tmp_path, arguments, expected):
    root = tmp_path / "install"
    coder = root / "profiles" / "coder"
    coder.mkdir(parents=True)
    (coder / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "config.yaml").write_text(
        "platforms:\n  webhook:\n    enabled: true\n", encoding="utf-8"
    )
    before = json.dumps({"notify": {"profile": "coder", "secret": "fixture-secret"}})
    (root / "webhook_subscriptions.json").write_text(before, encoding="utf-8")
    (coder / "cron").mkdir()
    (coder / "cron" / "jobs.json").write_text(json.dumps({"jobs": [
        {"id": "first", "name": "same"}, {"id": "second", "name": "same"},
    ]}), encoding="utf-8")
    env = dict(os.environ, HERMES_HOME=str(root))
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "webhook", "subscribe", "notify",
         *arguments],
        cwd=Path(__file__).resolve().parents[2], env=env,
        capture_output=True, text=True, timeout=40, check=False,
    )
    assert expected in result.stdout, result.stdout + result.stderr
    assert result.returncode != 0, result.stdout
    assert (root / "webhook_subscriptions.json").read_text(encoding="utf-8") == before


@pytest.mark.parametrize("remove", [False, True])
def test_concurrent_subscription_change_is_a_refusal_not_success(tmp_path, monkeypatch, capsys, remove):
    from hermes_cli import webhook

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "platforms:\n  webhook:\n    enabled: true\n", encoding="utf-8"
    )
    (tmp_path / "webhook_subscriptions.json").write_text(json.dumps({
        "notify": {"profile": "default", "secret": "fixture-secret"},
    }), encoding="utf-8")
    load = webhook._load_subscriptions

    def load_before_competing_write():
        snapshot = load()

        def competing_change(subscriptions):
            if remove:
                subscriptions.pop("notify")
            else:
                subscriptions["notify"]["enabled"] = False

        webhook._mutate_subscriptions(competing_change)
        return snapshot

    monkeypatch.setattr(webhook, "_load_subscriptions", load_before_competing_write)
    status = webhook.webhook_command(Namespace(
        webhook_action="subscribe", name="notify", secret="", prompt="", events="",
        description="stale edit", skills="", deliver="log", deliver_chat_id="",
    ))
    assert status == 1
    output = capsys.readouterr().out
    assert "changed concurrently" in output
    assert "Updated webhook subscription" not in output
    assert "Created webhook subscription" not in output
    assert load() == ({} if remove else {
        "notify": {"profile": "default", "secret": "fixture-secret", "enabled": False},
    })
