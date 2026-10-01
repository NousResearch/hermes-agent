"""Store submission control for the release pipeline.

The green run submits the verified ``.msixbundle`` with auto-publish off
(``targetPublishMode: "Manual"``). The submission API has no call that
releases a held submission, so publication never edits it: the publication
pass only checks its state and tells a person to click Publish now in Partner
Center once it is certified. Both entry points are safe to rerun.

Green run — ``python -m scripts.releases.store submit <package>`` (Windows
runner, ``msstore`` CLI configured beforehand by
``microsoft/microsoft-store-apppublisher`` + ``msstore reconfigure``):

1. ``msstore submission status <productId>`` — a submission is in flight when
   its status is not one of the terminal/failure states
   (``None, Published, PublishFailed, Canceled, CertificationFailed,
   PreProcessingFailed, CommitFailed``).
2. ``msstore submission delete <productId> --no-confirm`` — only when one is
   in flight. Deleting the last published submission would be wrong, so the
   status gates this.
3. ``msstore publish <package> --appId <productId> --noCommit`` — uploads the
   package and leaves the submission as a draft (``--noCommit`` / ``-nc``,
   "Disables committing the submission, keeping it in draft state").
4. ``msstore submission get <productId>`` — the complete submission JSON.
5. ``msstore submission update <productId> <json>`` — the same JSON with
   ``targetPublishMode: "Manual"``. For MSIX apps ``submission update`` sends
   the complete submission JSON, so it sets packages and publish mode at once.
6. ``msstore submission publish <productId>`` — commits; certification starts.

Publication pass — ``scripts.releases.store.check(...)`` (Ubuntu runner,
Partner Center submission REST API, stdlib ``urllib`` only, read-only):

1. ``POST https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token`` with
   ``grant_type=client_credentials`` and scope
   ``https://manage.devcenter.microsoft.com/.default`` (the same MS_STORE_*
   credentials the CLI uses).
2. ``GET https://manage.devcenter.microsoft.com/v1.0/my/applications/{id}`` —
   ``pendingApplicationSubmission.id`` names the held submission.
3. ``GET .../submissions/{id}/status``. ``Release`` (certified, held by
   Manual) and the certification states print a Publish now instruction as
   a GitHub warning; the live states are a no-op; a failed state raises.

Sources for the commands and fields above:
- msstore CLI commands and options (``submission status/get/update/delete
  --no-confirm/publish``, ``publish --noCommit``):
  https://learn.microsoft.com/en-us/windows/apps/publish/msstore-dev-cli/commands
- CLI CI/CD setup (``msstore reconfigure --tenantId --sellerId --clientId
  --clientSecret`` on the runner):
  https://learn.microsoft.com/en-us/windows/apps/publish/msstore-dev-cli/github-actions
- Submission resource fields (``targetPublishMode``:
  ``Immediate``/``Manual``/``SpecificDate``) and the status enum
  (``PendingCommit, CommitStarted, PreProcessing, Certification,
  CertificationFailed, Release, PendingPublication, Publishing, Published, ...``):
  https://learn.microsoft.com/en-us/windows/uwp/monetize/manage-app-submissions
- REST methods (get an app, get submission status):
  https://learn.microsoft.com/en-us/windows/uwp/monetize/get-an-app
  https://learn.microsoft.com/en-us/windows/uwp/monetize/get-status-for-an-app-submission
  The documented methods are get, create, update, commit, delete and status;
  none releases a held submission, and update refuses a committed one (409).
- Azure AD client-credentials token:
  https://learn.microsoft.com/en-us/windows/uwp/monetize/create-and-manage-submissions-using-windows-store-services#obtain-an-azure-ad-access-token
"""
from __future__ import annotations

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

# Submission statuses that are terminal or already decided; none of them is an
# in-flight submission the green run would need to clear, and none of them is
# worth touching at release time except the already-live set below.
_NOT_IN_FLIGHT = {
    "None", "Published", "PublishFailed", "Canceled",
    "CertificationFailed", "PreProcessingFailed", "CommitFailed",
}
_ALREADY_LIVE = {"PendingPublication", "Publishing", "Published"}
# The status of a submission that passed certification and is held by
# targetPublishMode Manual, waiting for release.
_CERTIFIED_HELD = "Release"


class StoreError(RuntimeError):
    """A Store CLI or API call failed. The calling run stays red."""


def _cli_run(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise StoreError(f"{' '.join(argv)} failed: {result.stderr.strip()}")
    return result.stdout


def _has_in_flight(status_output: str) -> bool:
    try:
        status = json.loads(status_output)
    except json.JSONDecodeError:
        raise StoreError(f"unreadable submission status: {status_output!r}")
    if not isinstance(status, dict) or "status" not in status:
        raise StoreError(f"unreadable submission status: {status_output!r}")
    return status["status"] not in _NOT_IN_FLIGHT


def submit(package: str, *, product_id: str, run=_cli_run) -> dict:
    """Submit ``package`` with auto-publish off, clearing any in-flight one."""
    if _has_in_flight(run(["msstore", "submission", "status", product_id])):
        run(["msstore", "submission", "delete", product_id, "--no-confirm"])
    run(["msstore", "publish", package, "--appId", product_id, "--noCommit"])
    submission = json.loads(run(["msstore", "submission", "get", product_id]))
    submission["targetPublishMode"] = "Manual"
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


# Statuses of a committed submission that has not been decided yet.
_IN_CERTIFICATION = {"CommitStarted", "PreProcessing", "Certification"}
_FAILED = {"CommitFailed", "PreProcessingFailed", "CertificationFailed", "PublishFailed",
           "Canceled"}


def check(*, product_id: str, tenant_id: str, client_id: str, client_secret: str,
          run=_http_run) -> str:
    """Report the held submission's state and what a person must do next.

    The submission API has no call that releases a held (Manual) submission,
    so the publication pass never edits or commits it: a certified submission
    is put live by "Publish now" in Partner Center. A failed submission leaves
    the run red.
    """
    token = _token(tenant_id, client_id, client_secret, run)

    def get(path: str) -> dict:
        response = run({"method": "GET", "url": API_ROOT + path,
                        "headers": {"Authorization": f"Bearer {token}"}})
        if response["status"] != 200:
            raise StoreError(f"GET {path} failed: {response['status']}")
        return response["body"]

    app = get(f"/applications/{product_id}")
    pending = (app.get("pendingApplicationSubmission") or {}).get("id")
    if not pending:
        _notice(f"No Store submission is pending for {product_id}. The green run submits "
                "one; check its stable-store job.")
        return "no-submission"
    state = get(f"/applications/{product_id}/submissions/{pending}/status").get("status")
    if state in _FAILED:
        raise StoreError(f"Store submission {pending} is {state}; resubmit from a green run")
    if state in _ALREADY_LIVE:
        return "already-live"
    if state == _CERTIFIED_HELD:
        _notice(f"Store submission {pending} passed certification and is held. "
                "Open it in Partner Center and click Publish now.")
        return "needs-publish-now"
    if state in _IN_CERTIFICATION:
        _notice(f"Store submission {pending} is still in certification ({state}). When it "
                "passes, open it in Partner Center and click Publish now.")
        return "in-certification"
    raise StoreError(f"Store submission {pending} has an unexpected status: {state!r}")


def _notice(text: str) -> None:
    # A GitHub annotation, so the manual step shows on the run's summary page.
    print(f"::warning title=Microsoft Store::{text}")


def check_from_env(env: dict, run=_http_run) -> str:
    """Check the Store submission when the environment configures it."""
    product_id = env.get("MS_STORE_PRODUCT_ID")
    if not product_id:
        print("Store check skipped: MS_STORE_PRODUCT_ID is not configured.",
              file=sys.stderr)
        return "not-configured"
    return check(
        product_id=product_id,
        tenant_id=env["MS_STORE_TENANT_ID"],
        client_id=env["MS_STORE_CLIENT_ID"],
        client_secret=env["MS_STORE_CLIENT_SECRET"],
        run=run,
    )


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "submit":
        product_id = os.environ["MS_STORE_PRODUCT_ID"]
        package = argv[1]
        result = submit(package, product_id=product_id)
        print(json.dumps(result, sort_keys=True))
        return 0
    if argv[:1] == ["check"]:
        print(json.dumps({"result": check_from_env(dict(os.environ))},
                         sort_keys=True))
        return 0
    print("usage: python -m scripts.releases.store submit <package> | check",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
