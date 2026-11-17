"""A real ``hermes chat -q`` turn using the hosted artifacts MCP server end to end.

The profile is signed in to Nous (temp ``HERMES_HOME``, fake Portal saying ``artifacts_access`` and
``paid_access`` are true), so Hermes attaches ``hermes-artifacts`` at ``artifacts.base_url`` with the
Nous bearer. The scripted model then: creates an artifact; updates it from a stale base (version 0),
which the server refuses with ``stale_base`` and the head version; retries from that head; and shows
the result. Assertions read what the model saw and, against the in-repo fake, the server's inbound
wire (tool names and arguments, the bearer on every request).

Every test runs twice: ``[fake]`` against the in-repo fake server and fake Portal, and ``[real]``
against the plane named by ``HERMES_ARTIFACTS_E2E_BASE_URL``, ``HERMES_ARTIFACTS_E2E_PORTAL_URL`` and
``HERMES_ARTIFACTS_E2E_TOKEN_FILE`` (or ``HERMES_ARTIFACTS_E2E_TOKEN``), skipped unless all three are set
(see README.md). The wire
assertions that need the fake's inbound log skip on ``[real]``.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from tests.e2e.core.hosted_artifacts._helpers import (
    ENV_BASE_URL,
    ENV_PORTAL_URL,
    ENV_TOKEN,
    ENV_TOKEN_FILE,
    TOOLS,
    FakeArtifactsServer,
    RealPlane,
    build_artifacts_home,
    configure,
    fake_portal,
    nous_token,
    offered,
    real_plane,
    refuses,
    tool,
)
from tests.e2e.core.mcp_plugins._helpers import FINAL, _tool_msgs_this_turn, call_tool, payload, provider, run_chat_q
from tests.e2e.core.mcp_plugins._plugin_helpers import reap_tagged
from tests.fakes.fake_llm_provider import Text

pytestmark = [
    pytest.mark.platforms("linux"),  # the orphan sweep reads /proc
    pytest.mark.live_system_guard_bypass,  # reap_tagged signals only this run's tagged tree
]

HTML_V1 = "<!doctype html><h1>Release notes</h1><p>DRAFT</p>"


def _unwrap(result: dict[str, Any]) -> dict[str, Any]:
    """The server's own JSON inside Hermes' ``{"result": ...}`` / ``{"error": ...}`` envelope; a tool
    error is marked ``_tool_error``."""
    if len(result) == 1:
        key, inner = next(iter(result.items()))
        if key in ("result", "error") and isinstance(inner, str):
            with contextlib.suppress(ValueError):
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    return {**parsed, "_tool_error": key == "error"}
    return result


def _results(body: dict[str, Any]) -> list[dict[str, Any]]:
    """What the model saw from each tool call this turn, as the server's JSON where it is JSON."""
    out = []
    for m in _tool_msgs_this_turn(body):
        content = m.get("content")
        text = content if isinstance(content, str) else "\n".join(p.get("text", "") for p in content or [])
        out.append(_unwrap(payload(text)))
    return out


def _edit_flow(record: dict[str, Any]):
    """create -> update from base 0 (stale) -> update from the reported head -> show -> answer."""
    body = record["body"]
    seen = _results(body)
    step = len(seen)
    if step == 0:
        return call_tool(body, tool("artifact_create"),
                         {"title": "Release notes", "files": [{"path": "index.html", "content": HTML_V1}]})
    artifact_id = seen[0].get("id")
    if not artifact_id:
        return Text(f"{FINAL} create failed: {seen[0]}")
    if step == 1:
        return call_tool(body, tool("artifact_update"), {
            "id": artifact_id, "base_version": 0,
            "edits": [{"path": "index.html", "old": "DRAFT", "new": "FINAL"}]})
    if step == 2:
        head = seen[1].get("head_version")
        if not isinstance(head, int):
            return Text(f"{FINAL} no head_version in the stale result: {seen[1]}")
        return call_tool(body, tool("artifact_update"), {
            "id": artifact_id, "base_version": head,
            "edits": [{"path": "index.html", "old": "DRAFT", "new": "FINAL"}]})
    if step == 3:
        return call_tool(body, tool("artifact_show"), {"id": artifact_id})
    return Text(FINAL)


@contextlib.contextmanager
def _plane(root: Path, tag: str, token: str, real: RealPlane | None) -> Iterator[tuple[str, str, Any]]:
    """``(artifacts base URL, portal URL, fake server or None)`` for this run."""
    if real is not None:
        yield real.base_url, real.portal_url, None
        return
    with fake_portal() as portal:
        server = FakeArtifactsServer(root, token, "http://portal.localhost", tag).start()
        try:
            yield server.base_url, portal.origin, (server, portal)
        finally:
            server.stop()


@pytest.fixture(scope="module", params=["fake", "real"])
def flow(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    real = real_plane() if request.param == "real" else None
    if request.param == "real" and real is None:
        pytest.skip(f"set {ENV_BASE_URL}, {ENV_PORTAL_URL} and {ENV_TOKEN_FILE} (or {ENV_TOKEN}) to run against a real plane")
    root = tmp_path_factory.mktemp(f"artifacts-flow-{request.param}")
    token = real.token if real else nous_token()
    with provider(_edit_flow) as llm:
        # The home (and its tag) exist before the plane so the fake server joins the reaped tree.
        eh = build_artifacts_home(root, llm.base_url)
        with _plane(root, eh.tag, token, real) as (base_url, portal_url, fake):
            configure(eh, base_url=base_url, portal_url=portal_url, token=token)
            try:
                proc = run_chat_q(eh, "Write the release notes as an artifact, fix DRAFT, then show it.")
            finally:
                reap_tagged(eh)
            mains = llm.main_requests()
        out: dict[str, Any] = {"proc": proc, "mains": mains, "token": token, "real": real,
                               "base_url": base_url}
        if fake is not None:
            server, portal = fake
            out.update(calls=server.tool_calls(), entries=server.entries(),
                       account_reads=portal.account_reads(), refreshes=portal.refreshes())
    assert proc.returncode == 0 and FINAL in proc.stdout, (
        f"turn did not complete: exit {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}")
    assert mains, "the model was never called"
    out["seen"] = _results(mains[-1])
    return out


def test_the_model_is_offered_the_hosted_artifacts_tools(flow: dict[str, Any]) -> None:
    first = offered(flow["mains"][0])
    missing = [t for t in TOOLS if tool(t) not in first]
    assert not missing, f"hosted artifacts tools not offered to the model: {missing}\n{first[:2000]}"


def test_a_stale_update_is_reported_to_the_model_and_the_retry_from_the_head_succeeds(flow: dict[str, Any]) -> None:
    seen = flow["seen"]
    assert len(seen) == 4, f"expected create, stale update, update, show; the model saw {seen}"
    created, stale, updated, shown = seen
    assert created.get("version") == 1 and created.get("id"), created
    assert stale.get("_tool_error") is True, f"the stale update did not reach the model as a tool error: {stale}"
    assert stale.get("error") == "stale_base", f"the stale update did not surface stale_base: {stale}"
    assert stale.get("head_version") == 1, f"the stale update did not carry head_version 1: {stale}"
    assert updated.get("id") == created["id"] and updated.get("version") == 2, updated
    assert shown.get("id") == created["id"] and shown.get("version") == 2, shown
    assert shown.get("viewer_url") and shown.get("title") == "Release notes", shown


def test_content_url_is_not_printed_to_the_terminal(flow: dict[str, Any]) -> None:
    urls = [r["content_url"] for r in flow["seen"] if isinstance(r.get("content_url"), str)]
    assert urls, f"no content_url in any tool result (vacuous): {flow['seen']}"
    leaked = [u for u in urls if u in flow["proc"].stdout]
    assert not leaked, f"content_url printed to stdout: {leaked}\n{flow['proc'].stdout[-2000:]}"


def _fake_only(flow: dict[str, Any]) -> None:
    if flow["real"] is not None:
        pytest.skip("wire log is only available from the in-repo fake server")


def test_the_server_received_the_scripted_calls_in_order(flow: dict[str, Any]) -> None:
    _fake_only(flow)
    calls = flow["calls"]
    names = [n for n, _ in calls]
    assert names == ["artifact_create", "artifact_update", "artifact_update", "artifact_show"], calls
    artifact_id = flow["seen"][0]["id"]
    assert calls[0][1]["title"] == "Release notes" and calls[0][1]["files"][0]["content"] == HTML_V1, calls[0]
    assert [a.get("base_version") for _, a in calls[1:3]] == [0, 1], calls
    assert all(a.get("id") == artifact_id for _, a in calls[1:]), calls


def test_every_request_to_the_server_carried_the_nous_bearer(flow: dict[str, Any]) -> None:
    _fake_only(flow)
    entries = flow["entries"]
    methods = [e["msg"].get("method") for e in entries if isinstance(e["msg"], dict)]
    assert "initialize" in methods and "tools/list" in methods, methods
    bearers = {e["bearer"] for e in entries}
    assert bearers == {flow["token"]}, f"requests carried {bearers}, not the Nous access token"


def test_the_entitlement_came_from_a_fresh_account_read_with_the_same_token(flow: dict[str, Any]) -> None:
    _fake_only(flow)
    reads = flow["account_reads"]
    assert reads, "Hermes attached the server without reading /api/oauth/account"
    assert {r.bearer for r in reads} == {flow["token"]}, reads
    assert not flow["refreshes"], f"a long-lived token was refreshed: {flow['refreshes']}"


def test_the_fake_server_is_stopped(flow: dict[str, Any]) -> None:
    _fake_only(flow)
    assert refuses(flow["base_url"]), f"the fake artifacts server still accepts connections at {flow['base_url']}"
