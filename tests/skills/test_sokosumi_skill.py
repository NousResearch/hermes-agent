"""Hermetic tests for the sokosumi skill helper.

Stdlib + pytest + unittest.mock only; NO live network. All HTTP goes through
sokosumi_api._urlopen, which every test replaces with a scripted fake. Fixture
payloads mirror the Sokosumi OpenAPI spec (v1): responses wrap in a
{data, meta} envelope, jobs use the 12-value lowercase status enum, tasks the
12-value UPPERCASE enum, job events the 6-value UPPERCASE enum.
"""

import io
import json
import re
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "autonomous-ai-agents"
    / "sokosumi"
)
sys.path.insert(0, str(SKILL_DIR / "scripts"))

import sokosumi_api as api

# ── Spec ground truth (from the Sokosumi OpenAPI v1 components) ─────────────

SPEC_JOB_STATUSES = {
    "started", "completed", "processing", "input_required", "result_pending",
    "failed", "payment_pending", "payment_failed", "refund_pending",
    "refund_resolved", "dispute_pending", "dispute_resolved",
}
SPEC_TASK_STATUSES = {
    "DRAFT", "READY", "INPUT_REQUIRED", "AUTHENTICATION_REQUIRED",
    "OUT_OF_CREDITS", "CREDITS_TOPPED_UP", "RUNNING", "AWAITING_EXTERNAL",
    "COMPLETED", "FAILED", "CANCEL_REQUESTED", "CANCELED",
}

# ── Fixture builders ────────────────────────────────────────────────────────


def envelope(data, next_cursor=None):
    meta = {"timestamp": "2021-01-01T00:00:00.000Z", "requestId": "req-1"}
    if next_cursor is not None:
        meta["pagination"] = {
            "cursor": None, "limit": 100, "total": 2, "nextCursor": next_cursor,
        }
    return {"data": data, "meta": meta}


def job_fixture(status="completed", result="# Result"):
    return {
        "id": "cmi4gmksz000104l8wps8p7fp",
        "createdAt": "2021-01-01T00:00:00.000Z",
        "updatedAt": "2021-01-01T00:00:00.000Z",
        "agentJobId": "ajob_123",
        "agentId": "agent_123",
        "agent": {"id": "agent_123", "name": "Research Agent"},
        "userId": "user_123",
        "user": {"id": "user_123"},
        "jobType": "PAID",
        "status": status,
        "credits": 25,
        "workspace": "ws_123",
        "events": [],
        "result": result if status == "completed" else None,
        "completedAt": "2021-01-01T00:10:00.000Z" if status == "completed" else None,
    }


INPUT_SCHEMA = {
    "input_data": [
        {
            "id": "question",
            "type": "string",
            "name": "Question",
            "data": {"description": "What to research"},
            "validations": [],
        }
    ]
}

UNAUTHORIZED_BODY = {
    "error": "Unauthorized",
    "message": "Authentication required",
    "meta": {"path": "/v1/users/me", "method": "GET"},
}


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, payload):
    return urllib.error.HTTPError(
        "https://api.sokosumi.com/v1/x", code, "err", {},
        io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


class FakeTransport:
    """Scripted _urlopen replacement: pops one canned response per request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


@pytest.fixture(autouse=True)
def _api_env(monkeypatch):
    monkeypatch.setenv("SOKOSUMI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("SOKOSUMI_API_URL", raising=False)


def transport(monkeypatch, responses):
    fake = FakeTransport(responses)
    monkeypatch.setattr(api, "_urlopen", fake)
    return fake


# ── request(): auth, envelope, errors, backoff ──────────────────────────────


class TestRequest:
    def test_bearer_auth_and_url(self, monkeypatch):
        fake = transport(monkeypatch, [envelope({"id": "user_123"})])
        api.request("GET", "users/me")
        req = fake.requests[0]
        assert req.full_url == "https://api.sokosumi.com/v1/users/me"
        assert req.get_header("Authorization") == "Bearer test-key-not-real"

    def test_env_base_url_override(self, monkeypatch):
        monkeypatch.setenv("SOKOSUMI_API_URL", "https://api.preprod.sokosumi.com/")
        fake = transport(monkeypatch, [envelope({})])
        api.request("GET", "users/me")
        assert fake.requests[0].full_url == "https://api.preprod.sokosumi.com/v1/users/me"

    def test_envelope_unwrapped(self, monkeypatch):
        transport(monkeypatch, [envelope({"id": "agent_123"})])
        assert api.request("GET", "agents/agent_123") == {"id": "agent_123"}

    def test_missing_key_never_calls_network(self, monkeypatch):
        monkeypatch.delenv("SOKOSUMI_API_KEY", raising=False)
        fake = transport(monkeypatch, [])
        with pytest.raises(api.ApiError) as exc:
            api.request("GET", "users/me")
        assert "app.sokosumi.com/connections" in str(exc.value)
        assert fake.requests == []

    def test_401_guidance_without_key_leak(self, monkeypatch):
        transport(monkeypatch, [http_error(401, UNAUTHORIZED_BODY)])
        with pytest.raises(api.ApiError) as exc:
            api.request("GET", "users/me")
        assert exc.value.status == 401
        assert "app.sokosumi.com/connections" in str(exc.value)
        assert "test-key-not-real" not in str(exc.value)

    def test_429_retried_with_backoff_then_succeeds(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(api.time, "sleep", sleeps.append)
        fake = transport(
            monkeypatch,
            [http_error(429, {}), http_error(429, {}), envelope({"ok": True})],
        )
        assert api.request("GET", "agents") == {"ok": True}
        assert len(fake.requests) == 3
        assert sleeps == [1, 2]

    def test_429_exhaustion_raises_after_retry_cap(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(api.time, "sleep", sleeps.append)
        fake = transport(monkeypatch, [http_error(429, {"message": "slow down"})] * 5)
        with pytest.raises(api.ApiError) as exc:
            api.request("GET", "agents")
        assert exc.value.status == 429
        assert "Rate limited" in str(exc.value)
        assert len(fake.requests) == api.MAX_429_RETRIES + 1
        assert sleeps == [1, 2, 4, 8]

    def test_non_json_success_body_raises_api_error(self, monkeypatch):
        class HtmlResponse(FakeResponse):
            def __init__(self):
                self._body = b"<html>gateway error</html>"

        monkeypatch.setattr(api, "_urlopen", lambda req, timeout=None: HtmlResponse())
        with pytest.raises(api.ApiError) as exc:
            api.request("GET", "users/me")
        assert "Non-JSON response" in str(exc.value)

    def test_non_429_error_not_retried(self, monkeypatch):
        fake = transport(monkeypatch, [http_error(500, {"message": "boom"})])
        with pytest.raises(api.ApiError):
            api.request("GET", "agents")
        assert len(fake.requests) == 1

    def test_query_params_and_repeated_category(self, monkeypatch):
        fake = transport(monkeypatch, [envelope([])])
        api.request("GET", "agents", query={"category": ["research", "code"], "limit": 5})
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(fake.requests[0].full_url).query)
        assert qs["category"] == ["research", "code"]
        assert qs["limit"] == ["5"]


# ── pagination ──────────────────────────────────────────────────────────────


class TestPaginate:
    def test_follows_next_cursor(self, monkeypatch):
        fake = transport(
            monkeypatch,
            [
                envelope([{"id": "agent_1"}], next_cursor="cmi4cursor"),
                envelope([{"id": "agent_2"}]),
            ],
        )
        items = api.paginate("agents")
        assert [i["id"] for i in items] == ["agent_1", "agent_2"]
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(fake.requests[1].full_url).query)
        assert qs["cursor"] == ["cmi4cursor"]

    def test_limit_stops_early(self, monkeypatch):
        fake = transport(monkeypatch, [envelope([{"id": "a"}, {"id": "b"}], next_cursor="x")])
        assert len(api.paginate("agents", limit=2)) == 2
        assert len(fake.requests) == 1

    def test_repeated_cursor_terminates(self, monkeypatch):
        # A server bug repeating the same nextCursor must not loop forever.
        fake = transport(
            monkeypatch,
            [
                envelope([{"id": "a"}], next_cursor="stuck"),
                envelope([{"id": "b"}], next_cursor="stuck"),
            ],
        )
        items = api.paginate("agents")
        assert [i["id"] for i in items] == ["a", "b"]
        assert len(fake.requests) == 2

    def test_limit_zero_rejected_by_parser(self, monkeypatch, capsys):
        transport(monkeypatch, [])
        with pytest.raises(SystemExit):
            api.main(["agents", "--limit", "0"])
        assert "must be >= 1" in capsys.readouterr().err


# ── hire(): the inputSchema echo contract ───────────────────────────────────


class TestHire:
    def test_schema_fetched_and_echoed_verbatim(self, monkeypatch):
        fake = transport(
            monkeypatch,
            [envelope(INPUT_SCHEMA), envelope(job_fixture(status="started", result=None))],
        )
        job = api.hire("agent_123", {"question": "How many planets?"}, 25, "My job")
        assert fake.requests[0].full_url.endswith("/v1/agents/agent_123/input-schema")
        post = fake.requests[1]
        assert post.full_url.endswith("/v1/agents/agent_123/jobs")
        body = json.loads(post.data.decode("utf-8"))
        assert body["inputSchema"] == INPUT_SCHEMA
        assert body["inputData"] == {"question": "How many planets?"}
        assert body["maxCredits"] == 25
        assert body["name"] == "My job"
        assert job["status"] == "started"

    def test_task_variant_posts_to_task_with_agent_id(self, monkeypatch):
        fake = transport(
            monkeypatch,
            [envelope(INPUT_SCHEMA), envelope(job_fixture(status="started", result=None))],
        )
        api.hire("agent_123", {"question": "q"}, None, None, task_id="tsk_123")
        post = fake.requests[1]
        assert post.full_url.endswith("/v1/tasks/tsk_123/jobs")
        body = json.loads(post.data.decode("utf-8"))
        assert body["agentId"] == "agent_123"
        assert "maxCredits" not in body and "name" not in body

    def test_transient_schema_fetch_422_retried_once(self, monkeypatch):
        # Observed live on preprod: GET input-schema intermittently 422s and
        # succeeds on immediate retry.
        fake = transport(
            monkeypatch,
            [
                http_error(422, {"message": "Failed to parse input schema"}),
                envelope(INPUT_SCHEMA),
                envelope(job_fixture(status="started", result=None)),
            ],
        )
        api.hire("agent_123", {"question": "q"}, None, None)
        assert len(fake.requests) == 3
        assert fake.requests[1].full_url.endswith("/v1/agents/agent_123/input-schema")

    def test_post_422_refetches_schema_and_retries_once(self, monkeypatch):
        changed_schema = {"input_data": [{"id": "question_v2", "type": "string",
                                          "name": "Question", "data": None,
                                          "validations": []}]}
        fake = transport(
            monkeypatch,
            [
                envelope(INPUT_SCHEMA),
                http_error(422, {"message": "input schema mismatch"}),
                envelope(changed_schema),
                envelope(job_fixture(status="started", result=None)),
            ],
        )
        api.hire("agent_123", {"question_v2": "q"}, None, None)
        assert len(fake.requests) == 4
        retry_body = json.loads(fake.requests[3].data.decode("utf-8"))
        assert retry_body["inputSchema"] == changed_schema

    def test_post_422_twice_raises(self, monkeypatch):
        fake = transport(
            monkeypatch,
            [
                envelope(INPUT_SCHEMA),
                http_error(422, {"message": "invalid input"}),
                envelope(INPUT_SCHEMA),
                http_error(422, {"message": "invalid input"}),
            ],
        )
        with pytest.raises(api.ApiError) as exc:
            api.hire("agent_123", {"wrong_field": "q"}, None, None)
        assert exc.value.status == 422
        assert len(fake.requests) == 4


# ── wait(): terminal detection across both vocabularies ────────────────────


class TestWait:
    @pytest.fixture(autouse=True)
    def _fast_clock(self, monkeypatch):
        monkeypatch.setattr(api.time, "sleep", lambda s: None)

    def test_job_polls_to_completed(self, monkeypatch, capsys):
        transport(
            monkeypatch,
            [
                envelope(job_fixture(status="started", result=None)),
                envelope(job_fixture(status="processing", result=None)),
                envelope(job_fixture(status="completed")),
            ],
        )
        assert api.wait("job", "job_123", interval=60, timeout=3600) == 0
        out = capsys.readouterr().out
        assert '"status": "completed"' in out

    def test_job_input_required_exits_2(self, monkeypatch):
        transport(monkeypatch, [envelope(job_fixture(status="input_required", result=None))])
        assert api.wait("job", "job_123", interval=60, timeout=3600) == 2

    def test_job_failure_states_exit_1(self, monkeypatch):
        for status in ("failed", "payment_failed", "refund_resolved", "dispute_resolved"):
            transport(monkeypatch, [envelope(job_fixture(status=status, result=None))])
            assert api.wait("job", "job_123", interval=60, timeout=3600) == 1

    def test_foreign_vocabulary_status_keeps_polling(self, monkeypatch):
        # RUNNING belongs to the event/task vocabularies, never job.status; if
        # it ever leaks into a job poll it must not match the job terminal or
        # blocked sets -- the loop keeps waiting for a real terminal status.
        transport(
            monkeypatch,
            [
                envelope(job_fixture(status="RUNNING", result=None)),
                envelope(job_fixture(status="result_pending", result=None)),
                envelope(job_fixture(status="completed")),
            ],
        )
        assert api.wait("job", "job_123", interval=60, timeout=3600) == 0

    def test_task_uppercase_terminal_and_blocked(self, monkeypatch):
        transport(monkeypatch, [envelope({"id": "tsk_123", "status": "COMPLETED"})])
        assert api.wait("task", "tsk_123", interval=60, timeout=3600) == 0
        transport(monkeypatch, [envelope({"id": "tsk_123", "status": "CANCELED"})])
        assert api.wait("task", "tsk_123", interval=60, timeout=3600) == 1
        for status in ("INPUT_REQUIRED", "AUTHENTICATION_REQUIRED", "OUT_OF_CREDITS"):
            transport(monkeypatch, [envelope({"id": "tsk_123", "status": status})])
            assert api.wait("task", "tsk_123", interval=60, timeout=3600) == 2

    def test_timeout_exits_1(self, monkeypatch):
        clock = iter([0, 0, 4000])
        monkeypatch.setattr(api.time, "monotonic", lambda: next(clock))
        transport(monkeypatch, [envelope(job_fixture(status="processing", result=None))] * 2)
        assert api.wait("job", "job_123", interval=60, timeout=3600) == 1


# ── status constants stay inside the spec enums ─────────────────────────────


class TestStatusVocabularies:
    def test_job_sets_are_spec_subsets(self):
        assert api.JOB_TERMINAL <= SPEC_JOB_STATUSES
        assert api.JOB_BLOCKED <= SPEC_JOB_STATUSES

    def test_task_sets_are_spec_subsets(self):
        assert api.TASK_TERMINAL <= SPEC_TASK_STATUSES
        assert api.TASK_BLOCKED <= SPEC_TASK_STATUSES

    def test_no_invented_statuses(self):
        # QUEUED never existed; RUNNING is never a job.status.
        assert "QUEUED" not in SPEC_JOB_STATUSES | SPEC_TASK_STATUSES
        assert "RUNNING" not in SPEC_JOB_STATUSES
        assert not (api.JOB_TERMINAL | api.JOB_BLOCKED) & {"QUEUED", "RUNNING"}


# ── whoami fallback, input flow, CLI surface ────────────────────────────────


class TestWhoami:
    def test_user_key(self, monkeypatch):
        transport(monkeypatch, [envelope({"id": "user_123"})])
        assert api.whoami() == {"kind": "user", "identity": {"id": "user_123"}}

    def test_coworker_key_falls_back(self, monkeypatch):
        transport(
            monkeypatch,
            [http_error(403, UNAUTHORIZED_BODY), envelope({"id": "cow_123"})],
        )
        assert api.whoami()["kind"] == "coworker"

    def test_both_failing_raises_original(self, monkeypatch):
        transport(
            monkeypatch,
            [http_error(401, UNAUTHORIZED_BODY), http_error(401, UNAUTHORIZED_BODY)],
        )
        with pytest.raises(api.ApiError) as exc:
            api.whoami()
        assert exc.value.status == 401


class TestCli:
    def test_input_request_and_provide_input(self, monkeypatch, capsys):
        fake = transport(
            monkeypatch,
            [envelope({"eventId": "event_123", "message": "How many?", "inputSchema": "s"})],
        )
        assert api.main(["input-request", "job_123"]) == 0
        assert json.loads(capsys.readouterr().out)["eventId"] == "event_123"
        assert fake.requests[0].full_url.endswith("/v1/jobs/job_123/input-request")

        fake = transport(monkeypatch, [envelope(job_fixture(status="processing", result=None))])
        rc = api.main(
            ["provide-input", "job_123", "--event-id", "event_123", "--input-json", '{"a": 1}']
        )
        assert rc == 0
        assert fake.requests[0].full_url.endswith("/v1/jobs/job_123/inputs")
        body = json.loads(fake.requests[0].data.decode("utf-8"))
        assert body == {"eventId": "event_123", "inputData": {"a": 1}}

    def test_malformed_input_json_rejected_by_parser(self, monkeypatch, capsys):
        transport(monkeypatch, [])
        with pytest.raises(SystemExit):
            api.main(["hire", "agent_123", "--input-json", "{not json"])
        assert "not valid JSON" in capsys.readouterr().err

    def test_create_task_draft_default_and_ready_flag(self, monkeypatch):
        fake = transport(monkeypatch, [envelope({"id": "tsk_123", "status": "DRAFT"})])
        assert api.main(["create-task", "--name", "Research"]) == 0
        assert fake.requests[0].full_url.endswith("/v1/tasks")
        assert json.loads(fake.requests[0].data.decode("utf-8"))["status"] == "DRAFT"

        fake = transport(monkeypatch, [envelope({"id": "tsk_123", "status": "READY"})])
        assert api.main(
            ["create-task", "--name", "Research", "--coworker-id", "cow_123", "--ready"]
        ) == 0
        body = json.loads(fake.requests[0].data.decode("utf-8"))
        assert body["status"] == "READY" and body["coworkerId"] == "cow_123"

    def test_coworkers_scope_capability_and_client_side_limit(self, monkeypatch, capsys):
        # GET /coworkers takes only scope/capability -- no limit/cursor params.
        fake = transport(
            monkeypatch,
            [envelope([{"id": f"cow_{i}", "capabilities": ["tasks"]} for i in range(3)])],
        )
        assert api.main(["coworkers", "--limit", "2", "--scope", "all",
                         "--capability", "tasks"]) == 0
        url = fake.requests[0].full_url
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert qs == {"scope": ["all"], "capability": ["tasks"]}
        assert len(json.loads(capsys.readouterr().out)) == 2

    def test_job_details_aggregates_and_tolerates_partial_failure(self, monkeypatch, capsys):
        transport(
            monkeypatch,
            [
                envelope(job_fixture()),
                envelope([{"id": "evt_1", "status": "COMPLETED"}]),
                envelope([{"id": "file_1", "status": "READY", "fileUrl": "https://x/f"}]),
                envelope([{"url": "https://example.com", "title": None}]),
                http_error(404, {"message": "no pending input request"}),
            ],
        )
        assert api.main(["job", "job_123", "--details"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["job"]["status"] == "completed"
        assert out["files"][0]["status"] == "READY"
        assert "input-request" in out["errors"]

    def test_api_error_prints_json_to_stderr_exit_1(self, monkeypatch, capsys):
        transport(monkeypatch, [http_error(401, UNAUTHORIZED_BODY)])
        assert api.main(["credits"]) == 1
        err = json.loads(capsys.readouterr().err)
        assert err["error"]["status"] == 401


# ── SKILL.md frontmatter honors the hardline description rule ───────────────


class TestSkillMd:
    def test_description_at_most_60_chars(self):
        text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        m = re.search(r"^description: (.*)$", text, re.MULTILINE)
        assert m, "frontmatter description missing"
        assert len(m.group(1)) <= 60, len(m.group(1))
        assert m.group(1).endswith(".")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
