"""The Store submission is held until publish and is safe to rerun."""
import json

import pytest


class _Cli:
    """Stands in for the msstore CLI. Records every argv it is handed."""

    def __init__(self, pending):
        self.events = []
        self.pending = pending

    def __call__(self, argv):
        name = " ".join(argv)
        if argv[:3] == ["msstore", "submission", "status"]:
            self.events.append(("status", name))
            return json.dumps(
                {"status": "Certification"} if self.pending
                else {"status": "Published"})
        if argv[:3] == ["msstore", "submission", "get"]:
            self.events.append(("get", name))
            return json.dumps({
                "targetPublishMode": "Immediate", "friendlyName": "Submission 2",
                "listings": {}})
        if argv[:3] == ["msstore", "submission", "delete"]:
            self.events.append(("delete-pending", name))
            return ""
        if argv[:3] == ["msstore", "submission", "update"]:
            self.events.append(("update", name))
            self.updated = json.loads(argv[4])
            return ""
        if argv[:3] == ["msstore", "submission", "publish"]:
            self.events.append(("commit", name))
            return ""
        if argv[:2] == ["msstore", "publish"]:
            self.events.append(("submit", name))
            return ""
        raise AssertionError(argv)

    def names(self):
        return [name for _kind, name in self.events]


def _index(names, prefix):
    return next(index for index, name in enumerate(names)
                if name.startswith(prefix))


def test_submit_deletes_the_in_flight_submission_first():
    from scripts.releases.store import submit

    cli = _Cli(pending=True)
    submit("verified/Store-1.2.3.msixbundle", product_id="9NTEST", run=cli)
    names = cli.names()
    assert _index(names, "msstore submission delete") \
        < _index(names, "msstore publish ")
    assert _index(names, "msstore submission delete") \
        < _index(names, "msstore submission publish")


def test_submit_skips_the_delete_when_nothing_is_in_flight():
    from scripts.releases.store import submit

    cli = _Cli(pending=False)
    submit("verified/Store-1.2.3.msixbundle", product_id="9NTEST", run=cli)
    assert not any(name.startswith("msstore submission delete")
                   for name in cli.names())


def test_submit_turns_auto_publish_off():
    from scripts.releases.store import submit

    cli = _Cli(pending=False)
    submit("verified/Store-1.2.3.msixbundle", product_id="9NTEST", run=cli)
    assert cli.updated["targetPublishMode"] == "Manual"


def test_submit_commits_the_held_submission():
    from scripts.releases.store import submit

    cli = _Cli(pending=False)
    submit("verified/Store-1.2.3.msixbundle", product_id="9NTEST", run=cli)
    names = cli.names()
    assert _index(names, "msstore submission update") \
        < _index(names, "msstore submission publish")
    assert names[-1].startswith("msstore submission publish")


def test_submit_keeps_the_submission_in_draft_until_the_mode_is_set():
    from scripts.releases.store import submit

    cli = _Cli(pending=False)
    submit("verified/Store-1.2.3.msixbundle", product_id="9NTEST", run=cli)
    submit_call = next(name for name in cli.names() if name.startswith("msstore publish"))
    assert "--noCommit" in submit_call
    assert _index(cli.names(), "msstore publish ") \
        < _index(cli.names(), "msstore submission update")


class _Api:
    """Stands in for the Partner Center submission REST API (read-only use)."""

    def __init__(self, status, pending="1152921504621243540"):
        self.requests = []
        self.status = status
        self.pending = pending

    def __call__(self, request):
        self.requests.append(request)
        url, method = request["url"], request["method"]
        if url.endswith("/token"):
            return {"status": 200, "body": {"access_token": "tok"}}
        if method != "GET":
            raise AssertionError(f"the check must never write: {method} {url}")
        if request.get("headers", {}).get("Authorization") != "Bearer tok":
            return {"status": 401, "body": {"code": "Unauthorized"}}
        if url.endswith("/applications/9NTEST"):
            app = {"id": "9NTEST"}
            if self.pending:
                app["pendingApplicationSubmission"] = {"id": self.pending}
            return {"status": 200, "body": app}
        if url.endswith(f"/submissions/{self.pending}/status"):
            return {"status": 200, "body": {"status": self.status}}
        raise AssertionError(request)


def _check(status, **kwargs):
    from scripts.releases.store import check

    api = _Api(status, **kwargs)
    result = check(product_id="9NTEST", tenant_id="T", client_id="C",
                   client_secret="S", run=api)
    return api, result


def test_a_certified_held_submission_asks_for_publish_now(capsys):
    _api, result = _check("Release")
    assert result == "needs-publish-now"
    out = capsys.readouterr().out
    assert out.startswith("::warning title=Microsoft Store::") and "Publish now" in out


@pytest.mark.parametrize("status", ["CommitStarted", "PreProcessing", "Certification"])
def test_a_submission_in_certification_says_what_comes_next(status, capsys):
    _api, result = _check(status)
    assert result == "in-certification"
    out = capsys.readouterr().out
    assert "Publish now" in out and status in out


@pytest.mark.parametrize("status", ["PendingPublication", "Publishing", "Published"])
def test_a_live_submission_is_a_noop(status, capsys):
    _api, result = _check(status)
    assert result == "already-live"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("status", ["CertificationFailed", "CommitFailed", "PublishFailed",
                                    "Canceled"])
def test_a_failed_submission_leaves_the_run_red_and_says_to_resubmit(status):
    from scripts.releases.store import StoreError

    with pytest.raises(StoreError, match="resubmit from a green run"):
        _check(status)


def test_an_unknown_submission_status_leaves_the_run_red():
    from scripts.releases.store import StoreError

    with pytest.raises(StoreError, match="unexpected status"):
        _check("SomethingNew")


def test_no_pending_submission_is_reported_not_invented(capsys):
    _api, result = _check("Release", pending=None)
    assert result == "no-submission"
    assert "::warning" in capsys.readouterr().out


def test_the_check_never_writes_to_the_store():
    # _Api raises on any non-GET after the token; every state must pass.
    for status in ("Release", "Certification", "Published"):
        api, _ = _check(status)
        assert all(r["method"] == "GET" for r in api.requests[1:])


def test_http_run_sends_the_request_headers(monkeypatch):
    import urllib.request

    from scripts.releases.store import _http_run

    sent = {}

    class _Response:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request):
        sent.update({key.lower(): value for key, value in request.header_items()})
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _http_run({"method": "GET", "url": "https://example.test/x",
               "headers": {"Authorization": "Bearer tok"}})
    assert sent["authorization"] == "Bearer tok"


def test_check_from_env_skips_when_the_store_is_not_configured():
    from scripts.releases.store import check_from_env

    assert check_from_env({}) == "not-configured"


def test_check_from_env_passes_the_configured_credentials():
    from scripts.releases.store import check_from_env

    api = _Api("Published")
    env = {
        "MS_STORE_PRODUCT_ID": "9NTEST",
        "MS_STORE_TENANT_ID": "TENANT",
        "MS_STORE_CLIENT_ID": "CLIENT",
        "MS_STORE_CLIENT_SECRET": "SECRET",
    }
    assert check_from_env(env, run=api) == "already-live"
    token = next(r for r in api.requests if r["url"].endswith("/token"))
    assert token["form"]["client_id"] == "CLIENT"
    assert token["form"]["client_secret"] == "SECRET"
