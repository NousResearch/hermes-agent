"""Store submission control for the release pipeline.

The green run submits the verified ``.msixbundle`` with auto-publish off
(``targetPublishMode: "Manual"``) and commits it, so certification runs before
the release gate and nothing a customer sees changes. The publication pass,
which runs only when the release really goes out, tries to switch that held
submission to ``Immediate`` and commit it again so it goes live by itself.
Microsoft documents no call that releases a held submission and does not say
whether update accepts a committed one; when it refuses (409), publication
prints a Publish now instruction instead, which is the manual step this
replaces. Both entry points are safe to rerun.

Green run — ``python -m scripts.releases.store submit <package>`` (Windows
runner, ``msstore`` CLI configured beforehand by
``microsoft/microsoft-store-apppublisher`` + ``msstore reconfigure``):

1. ``msstore submission get <productId>`` — the current submission as JSON.
2. ``msstore submission delete <productId> --no-confirm`` — for a draft,
   a failed submission, or a committed Manual submission whose recorded claim
   has a matching remote abandon marker. Other committed submissions stay
   protected; old submissions without an owner record need manual cleanup.
3. ``msstore publish <package> --appId <productId> --noCommit`` — uploads the
   package and leaves the submission as a draft (``--noCommit`` / ``-nc``,
   "Disables committing the submission, keeping it in draft state").
4. ``msstore submission get <productId>`` — the complete draft JSON.
5. ``msstore submission update <productId> <json>`` — the same JSON with
   ``targetPublishMode: "Manual"`` and the claim tag in ``notesForCertification``.
   Existing certification instructions are preserved. For MSIX apps the CLI
   sends the complete submission JSON, setting packages and publish mode at once.
6. ``msstore submission publish <productId>`` — commits; certification starts.

Publication pass — ``scripts.releases.store.release(...)`` (Ubuntu runner,
Partner Center submission REST API, stdlib ``urllib`` only):

1. ``POST https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token`` with
   ``grant_type=client_credentials`` and scope
   ``https://manage.devcenter.microsoft.com/.default`` (the same MS_STORE_*
   credentials the CLI uses).
2. ``GET https://manage.devcenter.microsoft.com/v1.0/my/applications/{id}`` —
   ``pendingApplicationSubmission.id`` names the held submission.
3. ``GET .../submissions/{id}`` — it must carry this release's Store bundle
   (``applicationPackages[].fileName``); a submission from a newer green run
   is left alone.
4. ``PUT .../submissions/{id}`` with ``targetPublishMode: "Immediate"``, then
   ``POST .../submissions/{id}/commit``. A 409 from either means the API will
   not change a committed submission: publication warns with the Publish now
   step and stays green. A live submission is a no-op.

Sources for the commands and fields above:
- msstore CLI commands and options (``submission get/update/publish/delete
  --no-confirm``, ``publish --noCommit``):
  https://learn.microsoft.com/en-us/windows/apps/publish/msstore-dev-cli/commands
- CLI CI/CD setup (``msstore reconfigure --tenantId --sellerId --clientId
  --clientSecret`` on the runner):
  https://learn.microsoft.com/en-us/windows/apps/publish/msstore-dev-cli/github-actions
- Submission resource fields (``targetPublishMode``:
  ``Immediate``/``Manual``/``SpecificDate``, ``applicationPackages``) and the
  status enum (``PendingCommit, CommitStarted, PreProcessing, Certification,
  CertificationFailed, Release, PendingPublication, Publishing, Published, ...``):
  https://learn.microsoft.com/en-us/windows/uwp/monetize/manage-app-submissions
- REST methods (get an app, get/update/commit a submission, get its status):
  https://learn.microsoft.com/en-us/windows/uwp/monetize/get-an-app
  https://learn.microsoft.com/en-us/windows/uwp/monetize/update-an-app-submission
  https://learn.microsoft.com/en-us/windows/uwp/monetize/commit-an-app-submission
  https://learn.microsoft.com/en-us/windows/uwp/monetize/get-status-for-an-app-submission
  Update documents 409 as "could not be updated because of the current state
  of the app" without naming the states.
- Azure AD client-credentials token:
  https://learn.microsoft.com/en-us/windows/uwp/monetize/create-and-manage-submissions-using-windows-store-services#obtain-an-azure-ad-access-token
"""
from __future__ import annotations

from collections.abc import Callable
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

API_ROOT = "https://manage.devcenter.microsoft.com/v1.0/my"
TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
TOKEN_SCOPE = "https://manage.devcenter.microsoft.com/.default"

_DRAFT = "PendingCommit"
_FAILED = {"CommitFailed", "PreProcessingFailed", "CertificationFailed", "PublishFailed",
           "ReleaseFailed", "Canceled"}
# Nothing pending a green run could clobber: the last submission is live or gone.
_SETTLED = {"None", "Published"}
# Committed and not live yet: certification, or certified and held.
_COMMITTED = {"CommitStarted", "PreProcessing", "Certification", "Release"}
_ALREADY_LIVE = {"PendingPublication", "Publishing", "Published"}


class StoreError(RuntimeError):
    """A Store CLI or API call failed. The calling run stays red."""


def _cli_run(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        raise StoreError(f"{' '.join(argv)} failed: {result.stderr.strip()}")
    return result.stdout


def _submission(submission_output: str) -> dict:
    """Read the submission ``msstore submission get`` prints.

    ``submission status`` is no use here: for a packaged (MSIX) app it prints a Spectre
    markup line (``Submission Status = ...``) and status tables, never JSON, and prints
    nothing at all when the app has no pending submission (msstore-cli
    Commands/Submission/StatusCommand.cs). ``submission get`` serializes the submission
    as JSON (GetCommand.cs), and it is what ``submit`` already reads below.
    """
    try:
        submission = json.loads(submission_output)
    except json.JSONDecodeError:
        raise StoreError(f"unreadable submission: {submission_output[:200]!r}")
    key = _key(submission, "status") if isinstance(submission, dict) else None
    if key is None:
        raise StoreError(f"unreadable submission: {submission_output[:200]!r}")
    return submission


def _key(submission: dict, name: str) -> str | None:
    """The submission's own spelling of ``name``. The CLI serializes .NET property names
    (``Status``, ``TargetPublishMode``); the REST API documents camelCase."""
    return next((k for k in submission if k.lower() == name.lower()), None)


def _notes(submission: dict) -> dict:
    raw = submission.get(_key(submission, "notesForCertification") or "notesForCertification") or ""
    try:
        notes = json.loads(raw)
    except (TypeError, ValueError):
        notes = None
    # Preserve existing certification instructions when adding our owner record.
    return notes if isinstance(notes, dict) else ({"notes": raw} if raw else {})


def _is_abandoned(claim_tag: str) -> bool:
    from scripts.releases.versioning import marker_ref, parse_attempt_ref

    parsed = parse_attempt_ref(claim_tag)
    if parsed is None:
        return False
    claim = f"refs/tags/{claim_tag}^{{}}"
    marker = f"refs/tags/{marker_ref(*parsed)}^{{}}"
    refs = {ref: sha for sha, ref in (line.split() for line in _cli_run(
        ["git", "ls-remote", "--tags", "origin", claim, marker]).splitlines())}
    # A similarly named or foreign marker must never authorize deleting a submission.
    return bool(refs.get(claim) and refs.get(marker) == refs[claim])


def submit(package: str, *, product_id: str, claim_tag: str | None = None,
           is_abandoned: Callable[[str], bool] | None = None, run=_cli_run) -> dict:
    """Submit held; replace a committed predecessor only after its claim was abandoned."""
    from scripts.releases.versioning import parse_attempt_ref

    if claim_tag is not None and parse_attempt_ref(claim_tag) is None:
        raise StoreError(f"invalid Store release claim: {claim_tag!r}")
    submission = _submission(run(["msstore", "submission", "get", product_id]))
    status = submission[_key(submission, "status")]
    owner = _notes(submission).get("claimTag")
    abandoned = (status in _COMMITTED
                 and submission.get(_key(submission, "targetPublishMode")) == "Manual"
                 and isinstance(owner, str) and parse_attempt_ref(owner) is not None
                 and (is_abandoned or _is_abandoned)(owner))
    if status == _DRAFT or status in _FAILED or abandoned:
        run(["msstore", "submission", "delete", product_id, "--no-confirm"])
    elif status not in _SETTLED:
        raise StoreError(f"a committed Store submission is {status}; rerun stable-store after "
                         "it is published, or after its owning claim is abandoned; a submission "
                         "without an owner record needs manual cleanup in Partner Center")
    run(["msstore", "publish", package, "--appId", product_id, "--noCommit"])
    submission = _submission(run(["msstore", "submission", "get", product_id]))
    submission[_key(submission, "targetPublishMode") or "TargetPublishMode"] = "Manual"
    notes = _notes(submission)
    if claim_tag is not None:
        notes["claimTag"] = claim_tag
    else:
        # The CLI may inherit the last submission's notes; that owner does not own this upload.
        notes.pop("claimTag", None)
    submission[_key(submission, "notesForCertification") or "notesForCertification"] = json.dumps(notes)
    run(["msstore", "submission", "update", product_id,
         json.dumps(submission, separators=(",", ":"))])
    run(["msstore", "submission", "publish", product_id])
    return {"productId": product_id, "targetPublishMode": "Manual"}


def _http_run(request: dict) -> dict:
    """Default injected runner: one HTTP request, stdlib only."""
    url = request["url"]
    data = None
    headers = {"Content-Type": "application/json", **request.get("headers", {})}
    body = request.get("body")
    if body is not None:
        data = json.dumps(body).encode()
    if "form" in request:
        data = urllib.parse.urlencode(request["form"]).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=request["method"])
    try:
        with urllib.request.urlopen(req) as response:
            raw = response.read().decode()
            return {"status": response.status,
                    "body": json.loads(raw) if raw else ""}
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"message": raw}
        return {"status": error.code, "body": parsed}


def _token(tenant_id: str, client_id: str, client_secret: str, run) -> str:
    response = run({"method": "POST", "url": TOKEN_URL.format(tenant=tenant_id),
                    "form": {
                        "grant_type": "client_credentials",
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scope": TOKEN_SCOPE,
                    }})
    if response["status"] != 200 or "access_token" not in response["body"]:
        raise StoreError(f"token request failed: {response['status']}")
    return response["body"]["access_token"]


class _Refused(StoreError):
    """The submission API answered 409: it will not change this submission now."""


def release(*, product_id: str, store_file: str, tenant_id: str, client_id: str,
            client_secret: str, run=_http_run) -> str:
    """Switch this release's held Store submission to ``Immediate`` and commit it.

    ``store_file`` is the Store bundle the release's candidate manifest names; a
    submission that does not carry it belongs to another green run and is left alone.
    """
    token = _token(tenant_id, client_id, client_secret, run)
    auth = {"Authorization": f"Bearer {token}"}

    def call(method: str, path: str, body=None) -> dict:
        request = {"method": method, "url": API_ROOT + path, "headers": auth}
        if body is not None:
            request["body"] = body
        response = run(request)
        if response["status"] == 409 and method != "GET":
            raise _Refused(f"{method} {path}: {response['body']}")
        if response["status"] != 200:
            raise StoreError(f"{method} {path} failed: {response['status']} {response['body']}")
        return response["body"]

    app = call("GET", f"/applications/{product_id}")
    pending = (app.get("pendingApplicationSubmission") or {}).get("id")
    if not pending:
        _notice(f"No Store submission is pending for {product_id}; check the release's stable-store job.")
        return "no-submission"
    path = f"/applications/{product_id}/submissions/{pending}"
    submission = call("GET", path)
    state = submission.get(_key(submission, "status") or "status")
    packages = submission.get(_key(submission, "applicationPackages") or "applicationPackages") or []
    bound = any(package.get("fileName") == store_file and package.get("fileStatus") != "PendingDelete"
                for package in packages)
    if state in _ALREADY_LIVE:
        return "already-live"
    if state in _FAILED:
        raise StoreError(f"Store submission {pending} is {state}; resubmit from a green run")
    if state != _DRAFT and state not in _COMMITTED:
        raise StoreError(f"Store submission {pending} has an unexpected status: {state!r}")
    if not bound:
        _notice(f"The pending Store submission {pending} does not carry {store_file}; a newer "
                "green run replaced it, and that release's publication releases it.")
        return "superseded"
    mode_key = _key(submission, "targetPublishMode") or "targetPublishMode"
    if submission.get(mode_key) == "Immediate" and state in _COMMITTED:
        return "already-immediate"
    submission[mode_key] = "Immediate"
    try:
        call("PUT", path, submission)
        call("POST", f"{path}/commit")
    except _Refused as refused:
        _notice(f"Store submission {pending} stays held: the submission API refused to switch it "
                f"to Immediate ({refused}). Once it is certified, select Publish now in Partner Center.")
        return "held"
    after = call("GET", f"{path}/status").get("status")
    if after in _FAILED:
        raise StoreError(f"Store submission {pending} is {after} after commit")
    return "immediate"


def _notice(text: str) -> None:
    # A GitHub annotation, so the manual step shows on the run's summary page.
    print(f"::warning title=Microsoft Store::{text}")


def store_bundle(manifest: dict) -> str:
    """The Store bundle a candidate manifest names (the one ``materialize --store-only`` picks)."""
    names = [item["path"] for item in manifest.get("files", [])
             if item["path"].startswith("Store-") and item["path"].endswith(".msixbundle")]
    if len(names) != 1:
        raise StoreError(f"the candidate manifest names {len(names)} Store bundles, not one")
    return names[0]


def release_from_env(env: dict, manifest: dict, run=_http_run) -> str:
    """Release the held Store submission for ``manifest`` when the environment configures the Store."""
    product_id = env.get("MS_STORE_PRODUCT_ID")
    if not product_id:
        print("Store release skipped: MS_STORE_PRODUCT_ID is not configured.",
              file=sys.stderr)
        return "not-configured"
    return release(
        product_id=product_id,
        store_file=store_bundle(manifest),
        tenant_id=env["MS_STORE_TENANT_ID"],
        client_id=env["MS_STORE_CLIENT_ID"],
        client_secret=env["MS_STORE_CLIENT_SECRET"],
        run=run,
    )


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "submit":
        product_id = os.environ["MS_STORE_PRODUCT_ID"]
        package = argv[1]
        result = submit(package, product_id=product_id, claim_tag=os.environ.get("HERMES_ARCHIVE_TAG"))
        print(json.dumps(result, sort_keys=True))
        return 0
    print("usage: python -m scripts.releases.store submit <package>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
