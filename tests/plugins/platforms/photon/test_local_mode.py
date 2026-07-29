from __future__ import annotations

import subprocess
import textwrap

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon.adapter import PhotonAdapter


def _local_config() -> PlatformConfig:
    return PlatformConfig(
        enabled=True,
        token="",
        extra={
            "local": True,
            "autostart_sidecar": False,
            "sidecar_port": 8789,
            "sidecar_token": "test-local-sidecar-token",
            "allowed_chat_ids": ["any;-;+155****0100"],
        },
    )


def _run_node(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "--input-type=module", "--eval", textwrap.dedent(script)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_local_mode_does_not_require_photon_cloud_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_PROJECT_ID", raising=False)
    monkeypatch.delenv("PHOTON_PROJECT_SECRET", raising=False)
    monkeypatch.delenv("PHOTON_LOCAL", raising=False)
    monkeypatch.setattr(
        photon_adapter, "load_project_credentials", lambda: (None, None)
    )

    cfg = _local_config()

    assert photon_adapter.validate_config(cfg) is True
    instance = PhotonAdapter(cfg)
    assert instance._local_mode is True
    assert instance._autostart_sidecar is False
    assert instance._sidecar_token == "test-local-sidecar-token"
    assert instance._allowed_chat_ids == {"any;-;+155****0100"}


def test_yaml_behavior_config_is_bridged_for_pre_adapter_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_names = [
        "PHOTON_LOCAL",
        "PHOTON_SIDECAR_URL",
        "PHOTON_SIDECAR_AUTOSTART",
        "PHOTON_ALLOWED_CHAT_IDS",
    ]
    for name in env_names:
        monkeypatch.delenv(name, raising=False)

    result = photon_adapter._apply_yaml_config(
        {},
        {
            "extra": {
                "local": True,
                "sidecar_url": "https://photon-sidecar.example.test",
                "autostart_sidecar": False,
                "allowed_chat_ids": ["chat-one", "chat-two"],
            }
        },
    )

    assert result is None
    assert photon_adapter.os.environ["PHOTON_LOCAL"] == "true"
    assert (
        photon_adapter.os.environ["PHOTON_SIDECAR_URL"]
        == "https://photon-sidecar.example.test"
    )
    assert photon_adapter.os.environ["PHOTON_SIDECAR_AUTOSTART"] == "false"
    assert photon_adapter._parse_chat_ids(
        photon_adapter.os.environ["PHOTON_ALLOWED_CHAT_IDS"]
    ) == {"chat-one", "chat-two"}
    monkeypatch.setattr(photon_adapter, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(photon_adapter.shutil, "which", lambda _name: None)
    assert photon_adapter.check_requirements() is True


def test_yaml_behavior_bridge_preserves_explicit_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_SIDECAR_URL", "https://explicit.example.test")

    photon_adapter._apply_yaml_config(
        {},
        {"extra": {"sidecar_url": "https://yaml.example.test"}},
    )

    assert (
        photon_adapter.os.environ["PHOTON_SIDECAR_URL"]
        == "https://explicit.example.test"
    )


def test_spectrum_runtime_selects_local_provider_without_cloud_credentials() -> None:
    result = _run_node(
        """
        import assert from "node:assert/strict";
        import { createSpectrumRuntime } from "./plugins/platforms/photon/sidecar/spectrum-runtime.mjs";

        const calls = [];
        let capturedOptions;
        const importer = async (specifier) => {
          calls.push(`import:${specifier}`);
          if (specifier === "@spectrum-ts/core") {
            return {
              Spectrum: async (options) => {
                capturedOptions = options;
                return { kind: "app" };
              },
              attachment() {}, voice() {}, text() {}, markdown() {}, typing() {},
            };
          }
          if (specifier === "@spectrum-ts/imessage-local") {
            const localIMessage = (app) => ({ provider: "local", app });
            localIMessage.config = () => ({ provider: "local" });
            return { localIMessage };
          }
          throw new Error(`unexpected import: ${specifier}`);
        };

        const runtime = await createSpectrumRuntime({
          localMode: true,
          projectId: undefined,
          projectSecret: undefined,
          telemetry: false,
          importer,
        });

        assert.equal(runtime.app.kind, "app");
        assert.deepEqual(runtime.imessage(runtime.app), {
          provider: "local",
          app: runtime.app,
        });
        assert.equal(capturedOptions.projectId, undefined);
        assert.equal(capturedOptions.projectSecret, undefined);
        assert.deepEqual(capturedOptions.providers, [{ provider: "local" }]);
        assert.equal(capturedOptions.platforms, undefined);
        assert.deepEqual(calls, [
          "import:@spectrum-ts/core",
          "import:@spectrum-ts/imessage-local",
        ]);
        """
    )
    assert result.returncode == 0, result.stderr


def test_spectrum_runtime_selects_cloud_provider_with_credentials() -> None:
    result = _run_node(
        """
        import assert from "node:assert/strict";
        import { createSpectrumRuntime } from "./plugins/platforms/photon/sidecar/spectrum-runtime.mjs";

        let capturedOptions;
        const importer = async (specifier) => {
          if (specifier === "@spectrum-ts/core") {
            return {
              Spectrum: async (options) => {
                capturedOptions = options;
                return { kind: "app" };
              },
              attachment() {}, voice() {}, text() {}, markdown() {}, typing() {},
            };
          }
          if (specifier === "spectrum-ts/providers/imessage") {
            const imessage = (app) => ({ provider: "cloud", app });
            imessage.config = () => ({ provider: "cloud" });
            return { imessage };
          }
          throw new Error(`unexpected import: ${specifier}`);
        };

        await createSpectrumRuntime({
          localMode: false,
          projectId: "project-id",
          projectSecret: "project-secret",
          telemetry: true,
          importer,
        });

        assert.equal(capturedOptions.projectId, "project-id");
        assert.equal(capturedOptions.projectSecret, "project-secret");
        assert.equal(capturedOptions.telemetry, true);
        assert.deepEqual(capturedOptions.providers, [{ provider: "cloud" }]);
        assert.equal(capturedOptions.platforms, undefined);
        """
    )
    assert result.returncode == 0, result.stderr


def test_sidecar_fans_out_to_multiple_gateway_consumers() -> None:
    result = _run_node(
        """
        import assert from "node:assert/strict";
        import { EventEmitter } from "node:events";
        import { createConsumerHub } from "./plugins/platforms/photon/sidecar/inbound-consumers.mjs";

        const makeConsumer = () => {
          const response = new EventEmitter();
          response.lines = [];
          response.write = (line) => {
            response.lines.push(line);
            return true;
          };
          return response;
        };

        const hub = createConsumerHub();
        const first = makeConsumer();
        const second = makeConsumer();
        hub.add(first);
        hub.add(second);

        await hub.deliver('{"messageId":"one"}');
        assert.deepEqual(first.lines, ['{"messageId":"one"}\\n']);
        assert.deepEqual(second.lines, ['{"messageId":"one"}\\n']);

        hub.remove(first);
        await hub.deliver('{"messageId":"two"}');
        assert.deepEqual(first.lines, ['{"messageId":"one"}\\n']);
        assert.deepEqual(second.lines, [
          '{"messageId":"one"}\\n',
          '{"messageId":"two"}\\n',
        ]);
        """
    )
    assert result.returncode == 0, result.stderr


def test_slow_consumer_does_not_block_healthy_consumer() -> None:
    result = _run_node(
        """
        import assert from "node:assert/strict";
        import { EventEmitter } from "node:events";
        import { createConsumerHub } from "./plugins/platforms/photon/sidecar/inbound-consumers.mjs";

        const makeConsumer = (backpressured = false) => {
          const response = new EventEmitter();
          response.lines = [];
          response.write = (line) => {
            response.lines.push(line);
            return !backpressured;
          };
          response.destroy = () => response.emit("close");
          return response;
        };

        const hub = createConsumerHub({ maxQueueLines: 4 });
        const stalled = makeConsumer(true);
        const healthy = makeConsumer(false);
        hub.add(stalled);
        hub.add(healthy);

        const outcome = await Promise.race([
          (async () => {
            await hub.deliver('{"messageId":"one"}');
            await hub.deliver('{"messageId":"two"}');
            return "delivered";
          })(),
          new Promise((resolve) => setTimeout(() => resolve("timed_out"), 100)),
        ]);

        assert.equal(outcome, "delivered");
        assert.deepEqual(healthy.lines, [
          '{"messageId":"one"}\\n',
          '{"messageId":"two"}\\n',
        ]);
        """
    )
    assert result.returncode == 0, result.stderr


def test_consumer_hub_buffers_without_blocking_when_disconnected() -> None:
    result = _run_node(
        """
        import assert from "node:assert/strict";
        import { EventEmitter } from "node:events";
        import { createConsumerHub } from "./plugins/platforms/photon/sidecar/inbound-consumers.mjs";

        const hub = createConsumerHub({ maxOrphanLines: 2 });
        const outcome = await Promise.race([
          hub.deliver('{"messageId":"one"}').then(() => "delivered"),
          new Promise((resolve) => setTimeout(() => resolve("timed_out"), 100)),
        ]);
        assert.equal(outcome, "delivered");

        const response = new EventEmitter();
        response.lines = [];
        response.write = (line) => {
          response.lines.push(line);
          return true;
        };
        hub.add(response);
        await new Promise((resolve) => setImmediate(resolve));

        assert.deepEqual(response.lines, ['{"messageId":"one"}\\n']);
        """
    )
    assert result.returncode == 0, result.stderr


def test_removing_backpressured_consumer_cleans_up_drain_wait() -> None:
    result = _run_node(
        """
        import assert from "node:assert/strict";
        import { EventEmitter } from "node:events";
        import { createConsumerHub } from "./plugins/platforms/photon/sidecar/inbound-consumers.mjs";

        const stalled = new EventEmitter();
        stalled.write = () => false;
        const hub = createConsumerHub();
        hub.add(stalled);
        await hub.deliver('{"messageId":"one"}');
        assert.equal(stalled.listenerCount("drain"), 1);

        hub.remove(stalled);
        await new Promise((resolve) => setImmediate(resolve));

        assert.equal(stalled.listenerCount("drain"), 0);
        assert.equal(stalled.listenerCount("close"), 0);
        assert.equal(stalled.listenerCount("error"), 0);
        """
    )
    assert result.returncode == 0, result.stderr


def test_consumer_queue_overflow_disconnects_only_stalled_consumer() -> None:
    result = _run_node(
        """
        import assert from "node:assert/strict";
        import { EventEmitter } from "node:events";
        import { createConsumerHub } from "./plugins/platforms/photon/sidecar/inbound-consumers.mjs";

        const stalled = new EventEmitter();
        stalled.write = () => false;
        let destroyed = 0;
        stalled.destroy = () => {
          destroyed += 1;
          stalled.emit("close");
        };

        const healthy = new EventEmitter();
        healthy.lines = [];
        healthy.write = (line) => {
          healthy.lines.push(line);
          return true;
        };

        const hub = createConsumerHub({ maxQueueLines: 1 });
        hub.add(stalled);
        hub.add(healthy);
        await hub.deliver('{"messageId":"one"}');
        await hub.deliver('{"messageId":"two"}');
        await new Promise((resolve) => setImmediate(resolve));

        assert.equal(destroyed, 1);
        assert.deepEqual(healthy.lines, [
          '{"messageId":"one"}\\n',
          '{"messageId":"two"}\\n',
        ]);
        """
    )
    assert result.returncode == 0, result.stderr


def test_consumer_queue_is_bounded_by_bytes() -> None:
    result = _run_node(
        """
        import assert from "node:assert/strict";
        import { EventEmitter } from "node:events";
        import { createConsumerHub } from "./plugins/platforms/photon/sidecar/inbound-consumers.mjs";

        const stalled = new EventEmitter();
        stalled.write = () => false;
        let destroyed = 0;
        stalled.destroy = () => {
          destroyed += 1;
          stalled.emit("close");
        };

        const hub = createConsumerHub({
          maxQueueLines: 100,
          maxQueueBytes: 10,
        });
        hub.add(stalled);
        await hub.deliver("12345");
        await hub.deliver("abcdef");
        await new Promise((resolve) => setImmediate(resolve));

        assert.equal(destroyed, 1);
        """
    )
    assert result.returncode == 0, result.stderr
