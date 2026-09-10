"""Contract tests for the route-script delivery-ID env var (HERMES_WEBHOOK_DELIVERY_ID).

Contract (nova dispatch 05a318b9 / thread thr_0f9279fca61f4884, Gatekeeper integration):

1. A route script receives the LITERAL platform delivery identifier — GitHub's
   ``X-GitHub-Delivery`` header value, byte-for-byte — as ``HERMES_WEBHOOK_DELIVERY_ID``.
2. Deliveries that send no identifier header leave the env var ABSENT (not fabricated:
   the epoch-ms dedup fallback must never leak into the script env). Scripts fail
   closed on absence.
3. Only this single whitelisted identifier is exposed: no other request headers and
   no Hermes-managed secrets appear in the script environment.
"""

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import (
    WebhookAdapter,
    _INSECURE_NO_AUTH,
)


def _make_adapter(routes=None, extra=None):
    _extra = extra or {}
    if routes:
        _extra["routes"] = routes
    _extra.setdefault("secret", "test-global-secret")
    config = PlatformConfig(enabled=True, extra=_extra)
    return WebhookAdapter(config)


def _create_app(adapter):
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _write_env_probe_script(scripts_dir, name="delivery_probe.py"):
    """Script that echoes the delivery-ID env var state back as its transformed payload."""
    script = scripts_dir / name
    script.write_text(
        "import json, os\n"
        "import sys\n"
        "payload = json.load(sys.stdin)\n"
        "payload['delivery_env'] = os.environ.get('HERMES_WEBHOOK_DELIVERY_ID', '<ABSENT>')\n"
        "payload['github_event_env'] = os.environ.get('HERMES_GITHUB_EVENT', '<ABSENT>')\n"
        "payload['signature_env'] = os.environ.get('HERMES_WEBHOOK_SIGNATURE', '<ABSENT>')\n"
        "print(json.dumps(payload))\n",
        encoding="utf-8",
    )
    return script


def _write_env_absent_fail_closed_script(scripts_dir, name="fail_closed_probe.py"):
    """Script implementing the fail-closed contract: no delivery ID -> drop the webhook."""
    script = scripts_dir / name
    script.write_text(
        "import json, os\n"
        "import sys\n"
        "if not os.environ.get('HERMES_WEBHOOK_DELIVERY_ID'):\n"
        "    sys.exit(3)  # fail closed: no delivery identity, refuse to process\n"
        "payload = json.load(sys.stdin)\n"
        "payload['seen_delivery_id'] = os.environ['HERMES_WEBHOOK_DELIVERY_ID']\n"
        "print(json.dumps(payload))\n",
        encoding="utf-8",
    )
    return script


class TestDeliveryIdEnvContract:
    """E2E through the real HTTP handler: header -> literal env var -> script."""

    @pytest.mark.asyncio
    async def test_github_delivery_id_reaches_script_as_literal_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        _write_env_probe_script(scripts)
        routes = {
            "gh": {
                "secret": _INSECURE_NO_AUTH,
                "script": "delivery_probe.py",
                "prompt": "Delivery {delivery_env}",
            }
        }
        adapter = _make_adapter(routes=routes)
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        literal_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/gh",
                json={"action": "opened", "pull_request": {"number": 42}},
                headers={"X-GitHub-Delivery": literal_id, "X-GitHub-Event": "pull_request"},
            )
            assert resp.status == 202

        await _drain(adapter)
        assert len(captured) == 1
        # The literal header value — byte-for-byte, no derivation — reached the script.
        assert captured[0].raw_message["delivery_env"] == literal_id
        assert captured[0].text == f"Delivery {literal_id}"

    @pytest.mark.asyncio
    async def test_svix_id_fallback_is_literal_too(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        _write_env_probe_script(scripts)
        routes = {
            "svix": {
                "secret": _INSECURE_NO_AUTH,
                "script": "delivery_probe.py",
                "prompt": "id={delivery_env}",
            }
        }
        adapter = _make_adapter(routes=routes)
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/svix",
                json={"type": "email.sent"},
                headers={"svix-id": "msg_svix_literal_99"},
            )
            assert resp.status == 202

        await _drain(adapter)
        assert len(captured) == 1
        assert captured[0].raw_message["delivery_env"] == "msg_svix_literal_99"

    @pytest.mark.asyncio
    async def test_no_id_header_leaves_env_absent_and_script_fails_closed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        _write_env_absent_fail_closed_script(scripts)
        routes = {
            "plain": {
                "secret": _INSECURE_NO_AUTH,
                "script": "fail_closed_probe.py",
                "prompt": "processed {seen_delivery_id}",
            }
        }
        adapter = _make_adapter(routes=routes)
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/plain",
                json={"event": "generic"},
                # NOTE: no X-GitHub-Delivery, no svix-id, no webhook-id, no X-Request-ID.
            )
            # The fail-closed script exited non-zero -> webhook ignored, 200 with status ignored.
            body = await resp.json()
            assert resp.status == 200
            assert body["status"] == "ignored"
            assert body["reason"] == "script"

        await _drain(adapter)
        assert captured == []

    @pytest.mark.asyncio
    async def test_fabricated_epoch_fallback_never_reaches_script_env(self, tmp_path, monkeypatch):
        """The dedup fallback (epoch ms) must not leak into HERMES_WEBHOOK_DELIVERY_ID."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        _write_env_probe_script(scripts)
        routes = {
            "nofab": {
                "secret": _INSECURE_NO_AUTH,
                "script": "delivery_probe.py",
                "prompt": "id={delivery_env}",
            }
        }
        adapter = _make_adapter(routes=routes)
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/webhooks/nofab", json={"event": "generic"})
            assert resp.status == 202

        await _drain(adapter)
        assert len(captured) == 1
        env_val = captured[0].raw_message["delivery_env"]
        # Absent marker from the probe script — NOT a 13-digit epoch-ms fabrication.
        assert env_val == "<ABSENT>"
        assert not (env_val.isdigit() and len(env_val) >= 12)

    @pytest.mark.asyncio
    async def test_only_delivery_id_exposed_no_other_headers_or_secrets(self, tmp_path, monkeypatch):
        """Whitelist contract: sibling headers and signature material never reach the script env."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        script = scripts / "env_dump_probe.py"
        script.write_text(
            "import json, os\n"
            "import sys\n"
            "payload = json.load(sys.stdin)\n"
            "hermes_owned = {k: v for k, v in os.environ.items() if k.startswith('HERMES_')}\n"
            "payload['hermes_owned_env'] = hermes_owned\n"
            "print(json.dumps(payload))\n",
            encoding="utf-8",
        )
        routes = {
            "sec": {
                "secret": _INSECURE_NO_AUTH,
                "script": "env_dump_probe.py",
                "prompt": "ok",
            }
        }
        adapter = _make_adapter(routes=routes)
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/sec",
                json={"event": "x"},
                headers={
                    "X-GitHub-Delivery": "literal-uuid-1",
                    "X-Hub-Signature-256": "sha256=deadbeef",
                    "X-GitHub-Event": "push",
                    "X-Custom-Secret": "topsecret",
                },
            )
            assert resp.status == 202

        await _drain(adapter)
        assert len(captured) == 1
        hermes_env = captured[0].raw_message["hermes_owned_env"]
        # Exactly one HERMES_* var is contract surface; anything else present must not
        # include header or signature material.
        assert hermes_env.get("HERMES_WEBHOOK_DELIVERY_ID") == "literal-uuid-1"
        dumped = json.dumps(hermes_env)
        assert "deadbeef" not in dumped
        assert "topsecret" not in dumped
        assert "X-Hub" not in dumped


class TestRunRouteScriptUnit:
    """Direct unit coverage of run_route_script's delivery-id parameter."""

    def _processor(self):
        from gateway.platforms.webhook_filters import WebhookRouteProcessor

        return WebhookRouteProcessor()

    def test_delivery_id_passed_as_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        script = scripts / "probe.py"
        script.write_text(
            "import json, os\n"
            "import sys\n"
            "print(json.dumps({'got': os.environ.get('HERMES_WEBHOOK_DELIVERY_ID', '<ABSENT>')}))\n",
            encoding="utf-8",
        )
        keep, transformed = self._processor().run_route_script("probe.py", {"x": 1}, "uuid-literal-7")
        assert keep is True
        assert transformed["got"] == "uuid-literal-7"

    def test_absent_delivery_id_leaves_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        script = scripts / "probe.py"
        script.write_text(
            "import json, os\n"
            "import sys\n"
            "print(json.dumps({'got': os.environ.get('HERMES_WEBHOOK_DELIVERY_ID', '<ABSENT>')}))\n",
            encoding="utf-8",
        )
        keep, transformed = self._processor().run_route_script("probe.py", {"x": 1}, None)
        assert keep is True
        assert transformed["got"] == "<ABSENT>"

    def test_legacy_two_arg_call_still_works(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        script = scripts / "probe.py"
        script.write_text(
            "import json, os\n"
            "import sys\n"
            "print(json.dumps({'got': os.environ.get('HERMES_WEBHOOK_DELIVERY_ID', '<ABSENT>')}))\n",
            encoding="utf-8",
        )
        # Backward compatibility: existing callers pass no delivery id.
        keep, transformed = self._processor().run_route_script("probe.py", {"x": 1})
        assert keep is True
        assert transformed["got"] == "<ABSENT>"

    def test_ambient_spoofed_env_is_scrubbed_when_no_delivery_id(self, tmp_path, monkeypatch):
        """Fail-closed scrub: the name is not a blocklisted secret, so an ambient value
        inherited by the gateway process would leak through build_subprocess_env() on
        no-ID deliveries. The runner must remove it — absence is the contract."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_WEBHOOK_DELIVERY_ID", "spoofed-ambient-value")
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        script = scripts / "probe.py"
        script.write_text(
            "import json, os\n"
            "import sys\n"
            "print(json.dumps({'got': os.environ.get('HERMES_WEBHOOK_DELIVERY_ID', '<ABSENT>')}))\n",
            encoding="utf-8",
        )
        keep, transformed = self._processor().run_route_script("probe.py", {"x": 1}, None)
        assert keep is True
        assert transformed is not None
        assert transformed["got"] == "<ABSENT>"

    def test_real_delivery_id_overrides_ambient_spoof(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_WEBHOOK_DELIVERY_ID", "spoofed-ambient-value")
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        script = scripts / "probe.py"
        script.write_text(
            "import json, os\n"
            "import sys\n"
            "print(json.dumps({'got': os.environ.get('HERMES_WEBHOOK_DELIVERY_ID', '<ABSENT>')}))\n",
            encoding="utf-8",
        )
        keep, transformed = self._processor().run_route_script("probe.py", {"x": 1}, "real-literal-id-9")
        assert keep is True
        assert transformed is not None
        assert transformed["got"] == "real-literal-id-9"

    def test_empty_string_delivery_id_treated_as_absent(self, tmp_path, monkeypatch):
        """An explicitly empty header value is not a delivery identity."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        script = scripts / "probe.py"
        script.write_text(
            "import json, os\n"
            "import sys\n"
            "print(json.dumps({'got': os.environ.get('HERMES_WEBHOOK_DELIVERY_ID', '<ABSENT>')}))\n",
            encoding="utf-8",
        )
        keep, transformed = self._processor().run_route_script("probe.py", {"x": 1}, "")
        assert keep is True
        assert transformed is not None
        assert transformed["got"] == "<ABSENT>"


async def _drain(adapter, seconds=0.35):
    import asyncio

    await asyncio.sleep(seconds)
    # Let any straggler background tasks settle without closing the per-delivery
    # session (which would cancel pending work).
    for _ in range(3):
        await asyncio.sleep(0.05)
