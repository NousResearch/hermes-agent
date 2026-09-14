"""The control plane's write path: who may act, what is recorded, and what is refused.

The audit that gated this phase put it in one sentence — *"an approval with no identity is
not an approval"* — and these tests are the other half of that: an identity that can act
without leaving a record is not an identity either.

The security tests carry the most weight. A loopback caller is a local admin, which was
harmless while everything was a read and is not harmless now, because a page in the
operator's own browser can make their browser POST to 127.0.0.1.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from nova.audit import AuditLog, NullAuditLog
from nova.control import ControlAPI
from nova.control.auth import Principal, PrincipalStore, WRITE_ROUTES, hash_token
from nova.control.server import MAX_BODY_BYTES, build_server
from nova.runtime.base import WORK_ACTIONS, AgentRuntime, WorkDecision

VIEWER = Principal(name="reader", role="viewer")
ADMIN = Principal(name="alice", role="admin")


class RecordingRuntime:
    """Stands in for a runtime, so the API's behaviour is tested without a board."""

    name = "recording"

    def __init__(self, *, applied=True, capable=True):
        from nova.runtime.base import RuntimeCapabilities

        self.capabilities = RuntimeCapabilities(
            work_decisions=capable, work_submission=True, channel_delivery=False
        )
        self.calls: list[dict] = []
        self._applied = applied

    def channel_readiness(self, channels, derivations=()):
        return [
            {"id": c.id, "provider": c.provider, "required_env": list(c.required_env),
             "missing_by_agent": {a: list(c.required_env) for a in c.allowed_agents},
             "ready": False}
            for c in channels
        ]

    def decide_work(self, task_id, action, *, actor, audit, correlation_id, reason="", note=""):
        self.calls.append(
            {"task_id": task_id, "action": action, "actor": actor, "reason": reason, "note": note}
        )
        if action == "annotate":
            with audit.model_visible_change(
                "work.annotated", correlation_id=correlation_id, subject=task_id,
                detail={"actor": actor},
            ):
                pass
        else:
            audit.record(
                "work.decided", correlation_id=correlation_id, subject=task_id,
                detail={"actor": actor, "action": action},
            )
        return WorkDecision(
            action=action, task_id=task_id, applied=self._applied,
            reason="" if self._applied else "already released",
            resulting_status="ready" if self._applied else "running",
        )


@pytest.fixture
def audit_log(tmp_path) -> AuditLog:
    return AuditLog(tmp_path / "audit.jsonl", tenant_id="acme", actor="nova")


@pytest.fixture
def write_api(bundle, audit_log):
    return ControlAPI(bundle, RecordingRuntime(), audit=audit_log)


# -- authorisation -----------------------------------------------------------


def test_a_viewer_may_not_act_on_work(write_api):
    response = write_api.write(
        "/platform/v1/work/t-1/decide", VIEWER, {"action": "release"}
    )
    assert response.status == 403
    assert "viewer" in response.body["error"]["message"]


def test_an_admin_may(write_api):
    response = write_api.write(
        "/platform/v1/work/t-1/decide", ADMIN, {"action": "release"}
    )
    assert response.status == 200
    assert response.body["applied"] is True
    assert response.body["actor"] == "alice"


def test_an_undeclared_write_route_is_a_404_not_a_403(write_api):
    """Forgetting to declare a read exposes data; forgetting to declare a write hands out
    an action. So an undeclared write is unroutable rather than admin-only.

    ``/agents/x/delete`` used to be the example and is now a real route, so the example
    moved to one that is still undeclared. Both halves are checked: an unknown collection,
    and a known collection with an action nobody declared — the second is the one a typo
    actually produces.
    """
    assert write_api.write("/platform/v1/knowledge/x/delete", ADMIN, {}).status == 404
    assert write_api.write("/platform/v1/agents/x/unpublish", ADMIN, {}).status == 404


def test_every_declared_write_route_names_a_real_role():
    from nova.control.auth import ROLES

    assert WRITE_ROUTES, "a write table with nothing in it means the phase did not land"
    for route, role in WRITE_ROUTES.items():
        assert role in ROLES, f"{route} requires {role!r}, which is not a role"


def test_a_read_route_cannot_be_reached_through_the_write_dispatcher(write_api):
    assert write_api.write("/platform/v1/budget", ADMIN, {}).status == 404


def test_the_write_dispatcher_is_not_the_read_dispatcher(write_api):
    """Separate entry points rather than one function branching on a method string,
    because a branch is a thing somebody eventually gets the wrong way round."""
    assert write_api.handle("/platform/v1/work/t-1/decide").status == 404


# -- the audit record --------------------------------------------------------


def test_a_decision_is_recorded_against_the_human_who_made_it(write_api, audit_log):
    write_api.write("/platform/v1/work/t-7/decide", ADMIN, {"action": "release"})
    events = [e for e in audit_log.read() if e.kind == "work.decided"]
    assert len(events) == 1
    assert events[0].detail["actor"] == "alice"
    assert events[0].subject == "t-7"


def test_a_note_is_write_ahead_because_a_worker_reads_it(write_api, audit_log):
    """A note becomes part of what a worker reads, exactly as a work item's body does.
    Releasing changes *when* a worker runs, not *what it reads* — hence the difference."""
    write_api.write(
        "/platform/v1/work/t-9/decide", ADMIN, {"action": "annotate", "note": "check refunds"}
    )
    phases = [e.phase for e in audit_log.read() if e.kind == "work.annotated"]
    assert phases == ["intent", "committed"]


def test_a_release_is_recorded_not_write_ahead(write_api, audit_log):
    write_api.write("/platform/v1/work/t-9/decide", ADMIN, {"action": "release"})
    events = [e for e in audit_log.read() if e.kind == "work.decided"]
    assert [e.phase for e in events] == ["record"]
    assert events[0].model_visible is False


def test_writes_are_refused_outright_with_no_audit_log(bundle):
    """A write path that can run without leaving a record is one somebody will run without
    leaving a record, and the record is the whole value of an approval."""
    api = ControlAPI(bundle, RecordingRuntime(), audit=None)
    response = api.write("/platform/v1/work/t-1/decide", ADMIN, {"action": "release"})
    assert response.status == 503
    assert "audit" in response.body["error"]["message"]


# -- refusals that are not errors --------------------------------------------


def test_a_decision_the_world_disagrees_with_is_409_not_400(bundle, audit_log):
    """An operator whose second click is told 'bad request' goes looking for a bug in the
    button. 409 says the request was fine and the world moved."""
    api = ControlAPI(bundle, RecordingRuntime(applied=False), audit=audit_log)
    response = api.write("/platform/v1/work/t-1/decide", ADMIN, {"action": "release"})
    assert response.status == 409
    assert response.body["applied"] is False
    assert response.body["reason"] == "already released"


def test_an_unknown_action_is_refused_with_the_list(write_api):
    response = write_api.write("/platform/v1/work/t-1/decide", ADMIN, {"action": "delete"})
    assert response.status == 400
    for action in WORK_ACTIONS:
        assert action in response.body["error"]["message"]


def test_a_runtime_that_cannot_act_says_so_rather_than_offering_a_dead_button(bundle, audit_log):
    api = ControlAPI(bundle, RecordingRuntime(capable=False), audit=audit_log)
    response = api.write("/platform/v1/work/t-1/decide", ADMIN, {"action": "release"})
    assert response.status == 501


def test_submitting_an_objective_that_does_not_exist_lists_the_ones_that_do(write_api):
    response = write_api.write("/platform/v1/objectives/nope/submit", ADMIN, {})
    assert response.status == 404
    assert "declared:" in response.body["error"]["message"]


# -- the transport -----------------------------------------------------------


@pytest.fixture
def live(bundle, audit_log):
    api = ControlAPI(bundle, RecordingRuntime(), audit=audit_log)
    server = build_server(api, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def post(url, body=None, *, content_type="application/json", origin=None, method="POST",
         raw=None, headers=None):
    data = raw if raw is not None else json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    if origin:
        req.add_header("Origin", origin)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_a_loopback_post_works(live):
    status, _ = post(f"{live}/platform/v1/work/t-1/decide", {"action": "release"})
    assert status == 200


def test_a_cross_site_origin_is_refused(live):
    """The attack this phase created and has to close: a page in the operator's browser
    making their browser POST to their own loopback admin."""
    status, body = post(
        f"{live}/platform/v1/work/t-1/decide",
        {"action": "release"},
        origin="https://evil.example",
    )
    assert status == 403
    assert b"cross-site" in body


def test_a_same_origin_header_is_accepted(live):
    host = live.split("//", 1)[1]
    status, _ = post(
        f"{live}/platform/v1/work/t-1/decide", {"action": "release"}, origin=f"http://{host}"
    )
    assert status == 200


def test_a_form_content_type_is_refused(live):
    """An HTML form can only send urlencoded, multipart or text/plain. Requiring JSON is
    what stops a form on another page from driving this API."""
    status, _ = post(
        f"{live}/platform/v1/work/t-1/decide",
        content_type="application/x-www-form-urlencoded",
        raw=b"action=release",
    )
    assert status == 415


def test_no_cors_preflight_is_answered(live):
    """Answering one is what would let a scripted cross-origin fetch send the real POST."""
    status, _ = post(f"{live}/platform/v1/work/t-1/decide", method="OPTIONS", raw=b"")
    assert status == 405


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
def test_other_write_methods_are_still_refused(live, method):
    status, _ = post(f"{live}/platform/v1/work/t-1/decide", {"action": "release"}, method=method)
    assert status == 405


def test_an_oversized_body_is_refused_before_it_is_read(live):
    status, _ = post(
        f"{live}/platform/v1/work/t-1/decide", raw=b"x" * (MAX_BODY_BYTES + 1)
    )
    assert status == 413


def test_a_malformed_body_says_so(live):
    status, body = post(f"{live}/platform/v1/work/t-1/decide", raw=b"{not json")
    assert status == 400
    assert b"valid JSON" in body


def test_a_non_object_body_is_refused(live):
    status, _ = post(f"{live}/platform/v1/work/t-1/decide", raw=b"[1,2,3]")
    assert status == 400


def test_a_post_outside_the_api_prefix_is_404(live):
    status, _ = post(f"{live}/index.html", {})
    assert status == 404


def test_a_remote_caller_without_a_token_cannot_write(bundle, audit_log, tmp_path):
    """The bind refusals already cover exposure; this covers the principal check itself."""
    store = PrincipalStore(principals=((hash_token("secret"), ADMIN),))
    api = ControlAPI(bundle, RecordingRuntime(), audit=audit_log)
    server = build_server(
        api, host="127.0.0.1", port=0, principals=store, behind_tls_proxy=True
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        # --behind-tls-proxy turns off loopback trust, so this is treated as remote.
        status, _ = post(f"http://{host}:{port}/platform/v1/work/t-1/decide", {"action": "release"})
        assert status == 401

        status, _ = post(
            f"http://{host}:{port}/platform/v1/work/t-1/decide",
            {"action": "release"},
            headers={"Authorization": "Bearer secret"},
        )
        assert status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# -- channels ----------------------------------------------------------------


def test_a_viewer_may_read_channels_but_not_apply_them(bundle, audit_log):
    """Reading what is connected is operational state and contains no credential. Making the
    runtime start delivering is a different permission, which is why the tables are separate."""
    from nova.control.auth import ROUTE_ROLES, WRITE_ROUTES

    assert ROUTE_ROLES["/channels"] == "viewer"
    assert WRITE_ROUTES["/channels/apply"] == "admin"

    api = ControlAPI(bundle, RecordingRuntime(), audit=audit_log)
    assert api.write("/platform/v1/channels/apply", VIEWER, {}).status == 403


def test_the_channels_read_route_never_returns_a_credential(bundle, audit_log, monkeypatch):
    """A control plane must be able to say a channel needs SLACK_BOT_TOKEN and must never be
    able to say what it is.

    This used to scan for the substring ``xoxb-``. That stopped working — and stopped
    testing anything — once the catalogue came from the runtime's own plugin manifests,
    because Slack's manifest writes its *format hint* as ``Slack Bot Token (xoxb-...)``.
    The hint is exactly the copy a connect form should show, so the check is now about
    values rather than about prefixes: a real token is planted in the environment, and the
    assertion is that no response ever carries it.
    """
    import json

    planted = "xoxb-9999-planted-secret-value-do-not-leak"
    monkeypatch.setenv("SLACK_BOT_TOKEN", planted)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", planted + "-telegram")

    api = ControlAPI(bundle, RecordingRuntime(), audit=audit_log)
    body = json.dumps(api.handle("/platform/v1/channels").body)

    assert planted not in body, "a credential VALUE reached the control plane's response"
    assert planted + "-telegram" not in body

    # A token-shaped string is one followed by token characters; the manifest's "(xoxb-...)"
    # is not. Keeping a shape check alongside the value check catches a leak from a source
    # the planting above did not anticipate.
    import re

    leaked = re.findall(r"xox[baprs]-[A-Za-z0-9]{4,}", body)
    assert not leaked, f"token-shaped strings in the response: {leaked}"

    # And the names must still be there, or the screen cannot tell an operator what to set.
    assert "SLACK_BOT_TOKEN" in body


def test_applying_channels_on_a_runtime_that_cannot_deliver_is_501(bundle, audit_log):
    """A connected-looking channel that silently goes nowhere is the worst failure this
    layer has, so the control plane refuses rather than offering the button."""
    api = ControlAPI(bundle, RecordingRuntime(), audit=audit_log)
    response = api.write("/platform/v1/channels/apply", ADMIN, {})
    assert response.status == 501
