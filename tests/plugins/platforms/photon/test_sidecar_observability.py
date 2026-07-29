"""Behavior tests for Photon sidecar stream observability helpers."""
from __future__ import annotations

import subprocess
import textwrap


def _run_node(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "--input-type=module", "--eval", textwrap.dedent(source)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_stream_health_snapshot_reports_degraded_duration() -> None:
    result = _run_node(
        """
        import assert from "node:assert/strict";
        import { buildStreamHealthSnapshot } from "./plugins/platforms/photon/sidecar/stream-observability.mjs";

        const snapshot = buildStreamHealthSnapshot(
          {
            state: "degraded",
            degradedSince: 1_000,
            lastHealthyAt: "healthy-at",
            lastIssueAt: "issue-at",
            lastIssue: "stream interrupted",
            issueCount: 3,
          },
          90_000,
          6_000,
        );
        assert.deepEqual(snapshot, {
          ok: false,
          state: "degraded",
          degradedForMs: 5_000,
          restartAfterMs: 90_000,
          lastHealthyAt: "healthy-at",
          lastIssueAt: "issue-at",
          lastIssue: "stream interrupted",
          issueCount: 3,
        });
        """
    )
    assert result.returncode == 0, result.stderr


def test_stream_classifier_intercepts_console_log_and_error() -> None:
    result = _run_node(
        """
        import assert from "node:assert/strict";
        import {
          classifyStreamLog,
          installStreamLogClassifier,
        } from "./plugins/platforms/photon/sidecar/stream-observability.mjs";

        const forwarded = [];
        const classified = [];
        const fakeConsole = {
          log: (...args) => forwarded.push(["log", ...args]),
          error: (...args) => forwarded.push(["error", ...args]),
        };
        const restore = installStreamLogClassifier(fakeConsole, (text) => {
          classified.push(classifyStreamLog(text));
        });

        fakeConsole.log("[spectrum.stream] stream interrupted; reconnecting");
        fakeConsole.error("[spectrum.stream] stream persistently failing");
        restore();
        fakeConsole.log("ordinary log");

        assert.deepEqual(classified, ["recovering", "degraded"]);
        assert.deepEqual(forwarded, [
          ["log", "[spectrum.stream] stream interrupted; reconnecting"],
          ["error", "[spectrum.stream] stream persistently failing"],
          ["log", "ordinary log"],
        ]);
        """
    )
    assert result.returncode == 0, result.stderr


def test_inbound_error_labels_cloud_catchup_failures_only() -> None:
    result = _run_node(
        """
        import assert from "node:assert/strict";
        import { inboundStreamErrorMessage } from "./plugins/platforms/photon/sidecar/stream-observability.mjs";

        const upstream = new Error("Unknown server error occurred");
        upstream.code = "internalError";
        upstream.cause = {
          path: "/photon.imessage.EventService/CatchUpEvents",
          details: "Unknown server error occurred",
        };
        const upstreamMessage = inboundStreamErrorMessage(upstream);
        assert.match(upstreamMessage, /upstream of Hermes/);
        assert.match(upstreamMessage, /Photon Spectrum CatchUpEvents/);

        const local = inboundStreamErrorMessage(new Error("authorization denied"));
        assert.match(local, /authorization denied/);
        assert.doesNotMatch(local, /upstream of Hermes/);
        """
    )
    assert result.returncode == 0, result.stderr
