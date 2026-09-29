"""Opt-in ``retry_on_script_failure``: a route script that FAILS (non-zero exit, timeout, missing script)
answers HTTP 503 + ``Retry-After`` and leaves the delivery ID unrecorded, so a sender with a persistent
retry queue (the OwnTracks app deletes a queued message on any 2xx) keeps the event and its retry is
processed once the script works again. A deliberate veto (exit 0, silent) and every route without the
opt-in keep the 200 ``ignored`` answer."""

import asyncio
import hashlib
import hmac
import json
import logging
from argparse import ArgumentParser
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import _DYNAMIC_ROUTES_FILENAME, _INSECURE_NO_AUTH, WebhookAdapter
from hermes_cli.subcommands.webhook import build_webhook_parser
from hermes_cli.webhook import webhook_command

_GOOD = "import json, sys\nprint(json.dumps(json.load(sys.stdin)))\n"
_FAILURES = {
    "nonzero_exit": "import sys\nsys.exit(3)\n",
    "missing_script": None,
    "timeout": "import time\ntime.sleep(60)\n",
}
_OPT_IN = {"retry_on_script_failure": True}
_FAILED = (503, {"status": "error", "reason": "script_failed", "route": "r"})
_IGNORED = (200, {"status": "ignored", "reason": "script", "route": "r"})


def _adapter(routes, **extra) -> WebhookAdapter:
    return WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": routes,
                                                              **extra}))


def _app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()  # the two ingress routes connect() registers
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    app.router.add_post("/p/{profile}/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _write_script(home, name: str, body) -> None:
    if body is None:
        return
    scripts = home / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / name).write_text(body, encoding="utf-8")


def _captured_runs(adapter: WebhookAdapter) -> list:
    seen = []

    async def _handle(event):
        seen.append(event)

    adapter.handle_message = _handle
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize("route_extra, script, expected", [
    pytest.param(_OPT_IN, _FAILURES["nonzero_exit"], _FAILED, id="opt-in-agent-route"),
    pytest.param({**_OPT_IN, "cron_job": "sweeper"}, _FAILURES["nonzero_exit"], _FAILED, id="opt-in-cron-job-route"),
    pytest.param({**_OPT_IN, "deliver_only": True, "deliver": "telegram"}, _FAILURES["nonzero_exit"], _FAILED,
                 id="opt-in-deliver-only-route"),
    pytest.param(_OPT_IN, "", _IGNORED, id="opt-in-veto-empty-stdout"),
    pytest.param(_OPT_IN, "print('[SILENT]')\n", _IGNORED, id="opt-in-veto-silent"),
    pytest.param(_OPT_IN, "print('{\"__hermes_ignore__\": true}')\n", _IGNORED, id="opt-in-veto-ignore-flag"),
    pytest.param(_OPT_IN, "print('{\"[SILENT]\": true}')\n", _IGNORED, id="opt-in-veto-silent-flag"),
    pytest.param({}, _FAILURES["nonzero_exit"], _IGNORED, id="default-nonzero-exit"),
    pytest.param({}, _FAILURES["missing_script"], _IGNORED, id="default-missing-script"),
    pytest.param({"retry_on_script_failure": False}, _FAILURES["nonzero_exit"], _IGNORED, id="explicit-false"),
])
async def test_script_outcome_decides_the_answer(route_extra, script, expected, tmp_path, monkeypatch):
    """Only a FAILED run on an opted-in route becomes 503, in every route mode and before the route acts; a
    veto stays 200 ``ignored`` with the opt-in, and a failure stays 200 ``ignored`` without it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_script(tmp_path, "hook.py", script)
    adapter = _adapter({"r": {"secret": _INSECURE_NO_AUTH, "script": "hook.py", **route_extra}})
    seen = _captured_runs(adapter)

    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/webhooks/r", json={"x": 1}, headers={"X-Request-ID": "d-1"})
        assert (resp.status, await resp.json()) == expected
        assert ("Retry-After" in resp.headers) is (resp.status == 503)
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", sorted(_FAILURES))
async def test_failed_script_answers_503_and_its_retry_is_processed_once_fixed(
        failure, tmp_path, monkeypatch, caplog):
    """The 503 must not consume the delivery ID: the sender's retry of the SAME delivery after the script is
    fixed runs normally instead of being answered ``duplicate``. The failure reaches WARNING (errors.log)
    without the payload, which for a location tracker is personal data."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_script(tmp_path, "owntracks.py", _FAILURES[failure])
    adapter = _adapter({"owntracks": {"secret": _INSECURE_NO_AUTH, "script": "owntracks.py", "prompt": "at {lat}",
                                      **_OPT_IN}}, script_timeout_seconds=2)
    seen = _captured_runs(adapter)
    post = {"json": {"lat": 51.123456}, "headers": {"X-Request-ID": "loc-1"}}

    with caplog.at_level(logging.INFO):
        async with TestClient(TestServer(_app(adapter))) as cli:
            failed = await cli.post("/webhooks/owntracks", **post)
            assert failed.status == 503
            assert (await failed.json())["reason"] == "script_failed"
            assert int(failed.headers["Retry-After"]) > 0
            assert seen == []

            _write_script(tmp_path, "owntracks.py", _GOOD)
            retried = await cli.post("/webhooks/owntracks", **post)
            assert retried.status == 202
            await asyncio.sleep(0.05)

    assert [event.text for event in seen] == ["at 51.123456"]
    assert any(r.levelno == logging.WARNING and r.name == "gateway.platforms.webhook" and "owntracks" in r.getMessage()
               for r in caplog.records)
    assert not [r for r in caplog.records if "51.123456" in r.getMessage()]


@pytest.mark.asyncio
async def test_profile_ingress_failure_answers_503_and_retry_runs_as_that_profile(tmp_path, monkeypatch):
    """``/p/<profile>/webhooks/<route>`` resolves the script under THAT profile's home: its failing copy answers
    503 although the launch profile's copy of the same name works, and the fixed retry runs as the profile."""
    worker = tmp_path / "profiles" / "worker"
    worker.mkdir(parents=True)
    (worker / "config.yaml").write_text("{}\n", encoding="utf-8")  # identity marker
    (worker / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: tmp_path / "profiles" / name)
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve",
                        lambda multiplex: [("default", tmp_path), ("worker", worker)])
    _write_script(tmp_path, "owntracks.py", _GOOD)
    _write_script(worker, "owntracks.py", _FAILURES["nonzero_exit"])
    adapter = _adapter({"owntracks": {"secret": _INSECURE_NO_AUTH, "profile": "worker", "script": "owntracks.py",
                                      **_OPT_IN}})
    adapter.gateway_runner = MagicMock()
    adapter.gateway_runner.config.multiplex_profiles = True
    seen = _captured_runs(adapter)
    post = {"json": {"lat": 1}, "headers": {"X-Request-ID": "loc-1"}}

    async with TestClient(TestServer(_app(adapter))) as cli:
        failed = await cli.post("/p/worker/webhooks/owntracks", **post)
        assert failed.status == 503
        assert "Retry-After" in failed.headers

        _write_script(worker, "owntracks.py", _GOOD)
        retried = await cli.post("/p/worker/webhooks/owntracks", **post)
        assert retried.status == 202
        await asyncio.sleep(0.05)

    assert [event.source.profile for event in seen] == ["worker"]


@pytest.mark.asyncio
async def test_cli_subscription_opt_in_reaches_the_dynamic_route(tmp_path, monkeypatch):
    """``hermes webhook subscribe --script ... --retry-on-script-failure`` writes a subscription the running
    adapter hot-reloads and honours: a signed POST to the dynamic route answers 503 on a failed script."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("platforms:\n  webhook:\n    enabled: true\n", encoding="utf-8")
    _write_script(tmp_path, "owntracks.py", _FAILURES["nonzero_exit"])
    parser = ArgumentParser()
    build_webhook_parser(parser.add_subparsers(dest="command"), cmd_webhook=webhook_command)
    args = parser.parse_args(["webhook", "subscribe", "owntracks", "--script", "owntracks.py",
                              "--retry-on-script-failure"])
    args.func(args)
    subs = json.loads((tmp_path / _DYNAMIC_ROUTES_FILENAME).read_text(encoding="utf-8"))
    secret = subs["owntracks"]["secret"]
    adapter = _adapter({})
    _captured_runs(adapter)
    body = b'{"lat": 1}'
    headers = {"Content-Type": "application/json",
               "X-Hub-Signature-256": "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()}

    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/webhooks/owntracks", data=body, headers=headers)
        assert resp.status == 503
        assert (await resp.json())["reason"] == "script_failed"
