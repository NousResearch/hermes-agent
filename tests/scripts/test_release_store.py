"""The green run commits the Store build held; publication tries to switch it to Immediate."""
import json

import pytest


class _Cli:
    """Stands in for the msstore CLI. Records every argv it is handed."""

    def __init__(self, status="Published", *, mode="Immediate", claim_tag=None):
        self.events = []
        self.status = status
        self.mode = mode
        self.claim_tag = claim_tag

    def __call__(self, argv):
        name = " ".join(argv)
        if argv[:3] == ["msstore", "submission", "status"]:
            # The real CLI prints Spectre markup for an MSIX app, never JSON, and nothing
            # when no submission is pending (rc.12-v0.21.7: "unreadable submission status: ''").
            self.events.append(("status", name))
            return f"Submission Status = {self.status}\n"
        if argv[:3] == ["msstore", "submission", "get"]:
            self.events.append(("get", name))
            uploaded = any(kind == "submit" for kind, _ in self.events)
            status = "PendingCommit" if uploaded else self.status
            # The CLI serializes .NET property names (rc.13-v0.21.7: '{"Id": ..., "ApplicationCategory": ...').
            return json.dumps({
                "Status": status, "TargetPublishMode": "Immediate" if uploaded else self.mode,
                "NotesForCertification": json.dumps({"claimTag": self.claim_tag})
                if self.claim_tag and not uploaded else "",
                "FriendlyName": "Submission 2", "Listings": {}})
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


def _names(cli):
    return cli.names()


def test_submit_commits_the_upload_held_so_certification_starts_before_the_gate():
    from scripts.releases.store import submit

    cli = _Cli()
    submit("verified/Store-1.2.3.msixbundle", product_id="9NTEST", run=cli)
    names = cli.names()
    upload = names.index("msstore publish verified/Store-1.2.3.msixbundle --appId 9NTEST --noCommit")
    update = next(i for i, n in enumerate(names) if n.startswith("msstore submission update 9NTEST "))
    assert upload < update < names.index("msstore submission publish 9NTEST") == len(names) - 1
    # The CLI's own spelling is kept, so update does not get two publish-mode keys.
    assert cli.updated["TargetPublishMode"] == "Manual" and "targetPublishMode" not in cli.updated


@pytest.mark.parametrize("status", ["PendingCommit", "CommitFailed", "CertificationFailed",
                                    "PreProcessingFailed", "ReleaseFailed", "Canceled"])
def test_submit_replaces_a_draft_or_failed_submission(status):
    from scripts.releases.store import submit

    cli = _Cli(status)
    submit("verified/Store-1.2.3.msixbundle", product_id="9NTEST", run=cli)
    names = cli.names()
    delete = names.index("msstore submission delete 9NTEST --no-confirm")
    assert delete < next(i for i, n in enumerate(names) if n.startswith("msstore publish "))


@pytest.mark.parametrize("status", ["Published", "None"])
def test_submit_leaves_a_settled_submission_alone(status):
    from scripts.releases.store import submit

    cli = _Cli(status)
    submit("verified/Store-1.2.3.msixbundle", product_id="9NTEST", run=cli)
    assert not any(name.startswith("msstore submission delete") for name in cli.names())


@pytest.mark.parametrize("status", ["CommitStarted", "Certification", "Release", "Publishing"])
def test_submit_refuses_to_clobber_an_earlier_release_on_its_way_live(status):
    from scripts.releases.store import StoreError, submit

    cli = _Cli(status)
    with pytest.raises(StoreError, match="rerun stable-store"):
        submit("verified/Store-1.2.3.msixbundle", product_id="9NTEST", run=cli)
    assert cli.names() == ["msstore submission get 9NTEST"]


@pytest.mark.parametrize("status", ["Certification", "Release"])
def test_submit_replaces_an_abandoned_claims_held_submission(status):
    from scripts.releases.store import submit

    prior_claim = "rc.1-v0.21.5"
    next_claim = "rc.2-v0.21.5"
    cli = _Cli(status, mode="Manual", claim_tag=prior_claim)
    checked = []

    def is_abandoned(claim_tag):
        checked.append(claim_tag)
        return claim_tag == prior_claim

    submit("verified/Store-next.msixbundle", product_id="9NTEST", claim_tag=next_claim,
           is_abandoned=is_abandoned, run=cli)

    names = cli.names()
    assert checked == [prior_claim]
    assert names.index("msstore submission delete 9NTEST --no-confirm") < names.index(
        "msstore publish verified/Store-next.msixbundle --appId 9NTEST --noCommit")
    assert names[-1] == "msstore submission publish 9NTEST"
    assert cli.updated["TargetPublishMode"] == "Manual"
    # Bind the new upload before committing so its own abandonment can be recovered.
    assert json.loads(cli.updated["NotesForCertification"])["claimTag"] == next_claim
    assert "notesForCertification" not in cli.updated


@pytest.mark.parametrize("status,mode,owner,marked_claim", [
    ("Release", "Manual", "rc.1-v0.21.5", None),
    ("Release", "Manual", "rc.1-v0.21.5", "rc.9-v0.21.5"),
    ("Release", "Manual", None, "rc.1-v0.21.5"),
    ("Release", "Manual", "v0.21.5", "v0.21.5"),
    ("Certification", "Immediate", "rc.1-v0.21.5", "rc.1-v0.21.5"),
    ("Release", "SpecificDate", "rc.1-v0.21.5", "rc.1-v0.21.5"),
    ("PendingPublication", "Manual", "rc.1-v0.21.5", "rc.1-v0.21.5"),
    ("Publishing", "Manual", "rc.1-v0.21.5", "rc.1-v0.21.5"),
])
def test_abandoned_cleanup_does_not_clobber_an_unproven_or_live_owner(status, mode, owner, marked_claim):
    from scripts.releases.store import StoreError, submit

    cli = _Cli(status, mode=mode, claim_tag=owner)
    with pytest.raises(StoreError):
        submit("verified/Store-next.msixbundle", product_id="9NTEST", claim_tag="rc.2-v0.21.5",
               is_abandoned=lambda claim_tag: claim_tag == marked_claim, run=cli)
    assert cli.names() == ["msstore submission get 9NTEST"]


STORE_FILE = "Store-HermesBundled-2026.6864.2132.0-win.msixbundle"


class _Api:
    """Stands in for the Partner Center submission REST API, with state."""

    def __init__(self, status="Release", files=(STORE_FILE,), pending="1152921504621243540",
                 commit_status=200, refuse_committed=False):
        self.requests = []
        self.status = status
        self.files = list(files)
        self.pending = pending
        self.commit_status = commit_status
        # Microsoft does not document whether update accepts a committed submission;
        # both answers are modelled.
        self.refuse_committed = refuse_committed
        self.mode = "Manual"
        self.commits = 0

    def __call__(self, request):
        self.requests.append(request)
        url, method = request["url"], request["method"]
        if url.endswith("/token"):
            return {"status": 200, "body": {"access_token": "tok"}}
        if request.get("headers", {}).get("Authorization") != "Bearer tok":
            return {"status": 401, "body": {"code": "Unauthorized"}}
        submission = f"/applications/9NTEST/submissions/{self.pending}"
        if method == "GET" and url.endswith("/applications/9NTEST"):
            app = {"id": "9NTEST"}
            if self.pending:
                app["pendingApplicationSubmission"] = {"id": self.pending}
            return {"status": 200, "body": app}
        if method == "GET" and url.endswith(submission):
            return {"status": 200, "body": {
                "id": self.pending, "status": self.status, "targetPublishMode": self.mode,
                "applicationPackages": [{"fileName": f, "fileStatus": "Uploaded"} for f in self.files]}}
        if method == "GET" and url.endswith(f"{submission}/status"):
            return {"status": 200, "body": {"status": self.status}}
        if method == "PUT" and url.endswith(submission):
            if self.status != "PendingCommit" and self.refuse_committed:
                return {"status": 409, "body": {"code": "InvalidState"}}
            self.mode = request["body"]["targetPublishMode"]
            return {"status": 200, "body": request["body"]}
        if method == "POST" and url.endswith(f"{submission}/commit"):
            if self.commit_status != 200:
                return {"status": self.commit_status, "body": {}}
            self.commits += 1
            self.status = "CommitStarted"
            return {"status": 200, "body": {"status": "CommitStarted"}}
        raise AssertionError(request)

    def writes(self):
        return [r for r in self.requests[1:] if r["method"] != "GET"]


def _release(api, store_file=STORE_FILE):
    from scripts.releases.store import release

    return release(product_id="9NTEST", store_file=store_file, tenant_id="T", client_id="C",
                   client_secret="S", run=api)


@pytest.mark.parametrize("status", ["Certification", "Release"])
def test_release_switches_this_releases_held_submission_to_immediate(status):
    api = _Api(status)
    assert _release(api) == "immediate"
    assert api.mode == "Immediate" and api.commits == 1
    put, commit = api.writes()
    assert put["method"] == "PUT" and commit["url"].endswith("/commit")


@pytest.mark.parametrize("refusal", ["update", "commit"])
def test_a_refused_switch_leaves_it_held_and_says_publish_now(capsys, refusal):
    api = _Api(refuse_committed=refusal == "update", commit_status=409 if refusal == "commit" else 200)
    assert _release(api) == "held"
    assert "Publish now" in capsys.readouterr().out


def test_release_is_safe_to_rerun():
    api = _Api()
    _release(api)
    assert _release(api) == "already-immediate"
    assert api.commits == 1


def test_release_leaves_a_submission_from_another_green_run_alone(capsys):
    api = _Api(files=["Store-HermesBundled-2026.7000.1.0-win.msixbundle"])
    assert _release(api) == "superseded"
    assert api.writes() == [] and "::warning" in capsys.readouterr().out


@pytest.mark.parametrize("status", ["PendingPublication", "Publishing", "Published"])
def test_a_live_submission_is_a_noop(status):
    api = _Api(status)
    assert _release(api) == "already-live" and api.writes() == []


@pytest.mark.parametrize("status", ["CertificationFailed", "CommitFailed", "PublishFailed", "Canceled"])
def test_a_failed_submission_raises_and_says_to_resubmit(status):
    from scripts.releases.store import StoreError

    with pytest.raises(StoreError, match="resubmit from a green run"):
        _release(_Api(status))


def test_any_other_api_failure_raises():
    from scripts.releases.store import StoreError

    with pytest.raises(StoreError, match="commit failed: 500"):
        _release(_Api(commit_status=500))


def test_no_pending_submission_is_reported_not_invented(capsys):
    api = _Api(pending=None)
    assert _release(api) == "no-submission"
    assert api.writes() == [] and "::warning" in capsys.readouterr().out


def test_an_unknown_submission_status_raises():
    from scripts.releases.store import StoreError

    with pytest.raises(StoreError, match="unexpected status"):
        _release(_Api("SomethingNew"))


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


def test_release_from_env_skips_when_the_store_is_not_configured():
    from scripts.releases.store import release_from_env

    assert release_from_env({}, {"files": []}) == "not-configured"


def test_release_from_env_releases_the_manifests_store_bundle_with_the_configured_credentials():
    from scripts.releases.store import release_from_env

    api = _Api()
    env = {
        "MS_STORE_PRODUCT_ID": "9NTEST",
        "MS_STORE_TENANT_ID": "TENANT",
        "MS_STORE_CLIENT_ID": "CLIENT",
        "MS_STORE_CLIENT_SECRET": "SECRET",
    }
    manifest = {"files": [{"path": "HermesBundled-0.22.0-win.msixbundle"}, {"path": STORE_FILE}]}
    assert release_from_env(env, manifest, run=api) == "immediate"
    token = next(r for r in api.requests if r["url"].endswith("/token"))
    assert token["form"]["client_id"] == "CLIENT"
    assert token["form"]["client_secret"] == "SECRET"


def test_submit_reads_the_status_as_the_submission_spells_it():
    """The CLI emits PascalCase; a camelCase (REST-shaped) submission works the same."""
    from scripts.releases.store import submit

    class _CamelCli(_Cli):
        def __call__(self, argv):
            out = super().__call__(argv)
            if argv[:3] == ["msstore", "submission", "get"]:
                data = json.loads(out)
                return json.dumps({k[0].lower() + k[1:]: v for k, v in data.items()})
            return out

    cli = _CamelCli("PendingCommit")
    submit("verified/Store-1.2.3.msixbundle", product_id="9NTEST", run=cli)
    assert "msstore submission delete 9NTEST --no-confirm" in cli.names()
