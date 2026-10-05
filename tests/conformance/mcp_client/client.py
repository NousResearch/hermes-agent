"""Client under test for the official MCP client conformance suite (``@modelcontextprotocol/conformance``).

The suite starts a scenario server and runs this script with the server URL as the last argument,
the scenario name in ``MCP_CONFORMANCE_SCENARIO`` and scenario data in ``MCP_CONFORMANCE_CONTEXT``.
It drives the code Hermes runs for a ``mcp_servers.<name>: {url: ...}`` entry: ``register_mcp_servers``
connects through ``MCPServerTask`` (Streamable HTTP, SSE fallback, schema cache, trust gates), the
registered tool is dispatched through ``tools.registry`` exactly as the model's call would be, and
``auth/*`` scenarios run the real ``HermesMCPOAuthProvider`` sign-in (discovery, CIMD/DCR, PKCE,
loopback callback listener, on-disk token storage under a throwaway ``HERMES_HOME``).

The browser is simulated: the authorization URL Hermes would open is fetched without following its
redirect, and the redirect is delivered to Hermes' own loopback callback server — the same HTTP
request a real browser makes. A user who keeps approving sign-ins would loop forever against a
server that never accepts the granted scope, so the simulated user gives up after three
(``auth/scope-retry-limit`` checks that bound).

Run through ``run.py``; that writes the outcome ``report`` consumed by the baseline comparison.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Every scenario gets a fresh home: the schema cache, token store and reserved callback ports of
# one scenario must not leak into the next (a cached token would skip the sign-in under test).
_HOME = Path(tempfile.mkdtemp(prefix="hermes-mcp-conformance-"))
os.environ["HOME"] = str(_HOME)
os.environ["HERMES_HOME"] = str(_HOME / ".hermes")
os.environ.pop("HERMES_SESSION_ID", None)
os.environ.pop("HERMES_YOLO_MODE", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("mcp_conformance.client")

SERVER_NAME = "conformance"
MAX_SIGN_INS = 3
REQUEST_TIMEOUT_SECONDS = 20.0
CALLBACK_DELIVERY_SECONDS = 10.0


def _report(path: str | None, payload: dict[str, Any]) -> None:
    if path:
        Path(path).write_text(json.dumps(payload, indent=2))


def _tool_calls(scenario: str, registered: list[str]) -> list[tuple[str, dict]] | None:
    """Tool calls of a scenario after connecting; ``None`` for scenarios this client does not know."""
    if scenario.startswith("auth/"):
        return [("test-tool", {})]
    if scenario == "initialize":
        return []
    if scenario == "tools_call":
        return [("add_numbers", {"a": 2, "b": 3})]
    if scenario == "sse-retry":
        return [("test_reconnection", {})]
    if scenario == "elicitation-sep1034-client-defaults":
        return [("test_client_elicitation_defaults", {})]
    if scenario == "json-schema-2020-12-preservation":
        # The server compares the echoed schema with the one it listed, to detect dropped keywords.
        from tools.mcp_tool_common import _core

        server = next(iter(_core._servers.values()), None)
        tool = next((t for t in (server._tools if server else []) if t.name == "json_schema_2020_12_tool"), None)
        if tool is None:
            raise RuntimeError("json_schema_2020_12_tool was not listed")
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)  # mcp 2.0 / 1.x
        return [("json_schema_echo", {"schema": schema})]
    return None


class SimulatedBrowser:
    """Fetches the authorization URL Hermes announces and delivers the redirect to Hermes' loopback
    callback listener, like a browser would; counts sign-ins so a server that never accepts the
    granted scope cannot loop the flow forever."""

    def __init__(self) -> None:
        self.sign_ins = 0
        self.failures: list[str] = []

    def announce(self, authorization_url: str, port: int, redirect_uri, redirect_host=None) -> None:
        self.sign_ins += 1
        if self.sign_ins > MAX_SIGN_INS:
            self.failures.append(f"sign-in requested {self.sign_ins} times; the simulated user gave up")
            # Delivering nothing lets the callback waiter time out, which ends the flow.
            return
        threading.Thread(target=self._drive, args=(authorization_url,), daemon=True).start()

    def _drive(self, authorization_url: str) -> None:
        import httpx

        try:
            response = httpx.get(authorization_url, follow_redirects=False, timeout=REQUEST_TIMEOUT_SECONDS)
            location = response.headers.get("location")
            if response.status_code not in (301, 302, 303, 307, 308) or not location:
                self.failures.append(f"authorization endpoint answered {response.status_code} without a redirect")
                return
            parsed = urlparse(location)
            if parsed.hostname not in ("127.0.0.1", "localhost") or not parsed.query:
                self.failures.append(f"authorization redirect does not target the loopback callback: {location}")
                return
            # Hermes binds the callback listener only after announcing the URL (the SDK calls the
            # redirect handler first); retry until it is listening.
            deadline = time.monotonic() + CALLBACK_DELIVERY_SECONDS
            while True:
                try:
                    httpx.get(location, timeout=5.0)
                    return
                except httpx.TransportError:
                    if time.monotonic() > deadline:
                        self.failures.append("loopback callback listener never came up")
                        return
                    time.sleep(0.1)
        except Exception as exc:  # the report names it; the suite's own checks tell the rest
            self.failures.append(f"simulated browser failed: {type(exc).__name__}: {exc}")


def _server_config(scenario: str, url: str) -> dict[str, Any]:
    config: dict[str, Any] = {"url": url, "enabled": True, "tool_timeout": REQUEST_TIMEOUT_SECONDS}
    if scenario.startswith("auth/"):
        config["auth"] = "oauth"
        oauth: dict[str, Any] = {"timeout": REQUEST_TIMEOUT_SECONDS}
        context = json.loads(os.environ.get("MCP_CONFORMANCE_CONTEXT") or "{}")
        if context.get("client_id"):  # auth/pre-registration hands the client its registration
            oauth["client_id"] = context["client_id"]
            if context.get("client_secret"):
                oauth["client_secret"] = context["client_secret"]
        if scenario == "auth/basic-cimd":
            # The scenario checks for this exact document URL (the reference clients hard-code it);
            # Hermes' own published document is what a real install sends.
            oauth["client_metadata_url"] = "https://conformance-test.local/client-metadata.json"
        config["oauth"] = oauth
    return config


def run(scenario: str, url: str) -> dict[str, Any]:
    import tools.mcp_tool  # noqa: F401 — populates the facade the siblings read through
    from tools import mcp_oauth
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.registry import registry

    browser = SimulatedBrowser()
    mcp_oauth._announce_authorization_url = browser.announce
    # The elicitation scenario's user accepts the form (SEP-1034: Hermes then answers with the schema defaults).
    import tools.approval_prompt as approval_prompt

    approval_prompt.request_elicitation_consent = lambda *args, **kwargs: "accept"
    outcome: dict[str, Any] = {"scenario": scenario, "ok": False, "sign_ins": 0, "errors": []}
    try:
        with mcp_oauth.force_interactive_oauth():
            registered = register_mcp_servers({SERVER_NAME: _server_config(scenario, url)})
            logger.info("registered %s", registered)
            if not registered and scenario != "initialize":
                from tools.mcp_tool_common import _core

                server = next(iter(_core._servers.values()), None)
                outcome["errors"].append(f"no tools registered: {getattr(server, '_error', None)!r}")
            calls = _tool_calls(scenario, registered)
            if calls is None:
                outcome["errors"].append(f"unknown scenario {scenario!r}")
            for tool, args in calls or []:
                from tools.mcp_tool_schema import mcp_prefixed_tool_name

                result = registry.dispatch(mcp_prefixed_tool_name(SERVER_NAME, tool), args)
                logger.info("%s(%s) -> %s", tool, args, str(result)[:300])
                if isinstance(result, str) and result.startswith("{\"error\"") and "isError" not in result:
                    try:
                        error = json.loads(result).get("error")
                    except ValueError:
                        error = result
                    if isinstance(error, str) and error.startswith("Unknown tool"):
                        outcome["errors"].append(error)
        outcome["ok"] = not outcome["errors"] and not browser.failures
    except Exception as exc:
        outcome["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        outcome["sign_ins"] = browser.sign_ins
        outcome["errors"].extend(browser.failures)
        try:
            shutdown_mcp_servers(timeout=5.0)
        except Exception as exc:  # noqa: BLE001 — teardown must not mask the scenario outcome
            logger.warning("shutdown failed: %s", exc)
    return outcome


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: client.py <server-url>", file=sys.stderr)
        return 2
    url = sys.argv[-1]
    scenario = os.environ.get("MCP_CONFORMANCE_SCENARIO", "")
    outcome = run(scenario, url)
    _report(os.environ.get("HERMES_MCP_CONFORMANCE_REPORT"), outcome)
    logger.info("outcome %s", json.dumps(outcome))
    shutil.rmtree(_HOME, ignore_errors=True)
    return 0 if outcome["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
