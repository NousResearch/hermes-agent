#!/usr/bin/env python3
"""Check OAuth token health for Google Workspace and Outlook MCP via live probes.

Outputs JSON summary for cron ingestion. Falls back to age-heuristic if the
live probe can't run (missing deps, file shape changed, transient network).
"""

import json
import os
import sys
import stat
from pathlib import Path
from datetime import datetime, timezone

GOOGLE_TOKEN_DIR = Path(os.path.expanduser("~/.google_workspace_mcp/credentials"))
OUTLOOK_CACHE = Path(os.path.expanduser("~/.config/ms-365-mcp-server/token-cache.json"))

NOW = datetime.now(timezone.utc)
HTTP_TIMEOUT = 10


def age_status(path):
    """Fallback: classify by file mtime."""
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = (NOW - mtime).days
    status = "healthy" if age < 5 else "warning" if age < 7 else "expired"
    return age, status


def probe_google(token_file):
    """Refresh creds and hit userinfo. Returns (status, detail)."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        import requests as rq
    except ImportError as exc:
        return "unknown", f"deps missing: {exc}"

    try:
        creds = Credentials.from_authorized_user_file(str(token_file))
    except Exception as exc:
        return "expired", f"creds parse failed: {exc}"

    try:
        creds.refresh(Request())
    except Exception as exc:
        return "expired", f"refresh failed: {exc}"

    try:
        r = rq.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {creds.token}"},
            timeout=HTTP_TIMEOUT,
        )
    except rq.RequestException as exc:
        return "unknown", f"network: {exc}"

    if r.status_code == 200:
        return "healthy", "userinfo 200"
    if r.status_code == 401:
        return "expired", "userinfo 401"
    return "warning", f"userinfo {r.status_code}"


def refresh_outlook_token(refresh_token, client_id, realm, scope):
    """Exchange refresh_token for a fresh access_token via Microsoft OAuth."""
    try:
        import requests as rq
    except ImportError:
        return None, "requests missing"

    url = f"https://login.microsoftonline.com/{realm}/oauth2/v2.0/token"
    try:
        r = rq.post(
            url,
            data={
                "client_id": client_id,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
                "scope": scope,
            },
            timeout=HTTP_TIMEOUT,
        )
    except rq.RequestException as exc:
        return None, f"network: {exc}"

    if r.status_code == 200:
        return r.json().get("access_token"), "refreshed"
    # Don't echo r.text into the report — response bodies on auth-error paths can include
    # request ids, scope fragments, or other identifiers that aren't meant to surface in cron logs.
    return None, f"refresh {r.status_code}"


def probe_outlook_account(access_token, account_has_user_read=False):
    """Probe Outlook account health graph.microsoft.com/v1.0/me or functional endpoints."""
    try:
        import requests as rq
    except ImportError as exc:
        return "unknown", f"deps missing: {exc}"

    # Accounts without User.Read (e.g. --preset mail,calendar) will 403 on /me.
    # Their mail and calendar endpoints remain functional.
    # Probe a working endpoint instead based on available scopes.
    if not account_has_user_read:
        try:
            r = rq.get(
                "https://graph.microsoft.com/v1.0/me/messages?$top=1",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=HTTP_TIMEOUT,
            )
        except rq.RequestException as exc:
            return "unknown", f"network: {exc}"

        if r.status_code == 200:
            return "healthy", "mail functional (no User.Read /me 403 expected)"
        if r.status_code == 401:
            return "expired", f"mail 401"
        return "warning", f"mail {r.status_code}"

    try:
        r = rq.get(
            "https://graph.microsoft.com/v1.0/me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=HTTP_TIMEOUT,
        )
    except rq.RequestException as exc:
        return "unknown", f"network: {exc}"

    if r.status_code == 200:
        return "healthy", "/me 200"
    if r.status_code == 401:
        return "expired", "/me 401"
    return "warning", f"/me {r.status_code}"


def _assert_secure_perms(path: Path) -> None:
    """Raise if a credential file is group- or world-readable."""
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(
            f"{path} has insecure permissions {oct(mode & 0o777)}; expected 0o600 or stricter"
        )


def parse_outlook_cache(cache_path):
    """Yield account info per Outlook account in the MSAL cache.

    Each yielded dict has: email, access_token, expires_on, refresh_token,
    client_id, realm, scope, has_user_read (best-effort).
    """
    _assert_secure_perms(cache_path)
    envelope = json.loads(cache_path.read_text())
    inner = envelope.get("data")
    if isinstance(inner, str):
        inner = json.loads(inner)
    accounts = inner.get("Account", {}) or {}
    access_tokens = inner.get("AccessToken", {}) or {}
    refresh_tokens = inner.get("RefreshToken", {}) or {}

    home_to_email = {}
    for acct in accounts.values():
        home_id = acct.get("home_account_id")
        email = acct.get("username") or acct.get("local_account_id")
        if home_id and email:
            home_to_email[home_id] = email

    home_to_best_at = {}
    for tok in access_tokens.values():
        home_id = tok.get("home_account_id")
        secret = tok.get("secret")
        if not (home_id and secret):
            continue
        expires_on = int(tok.get("expires_on", 0) or 0)

        # Check if User.Read is in the scopes
        target = tok.get("target", "")
        if isinstance(target, str):
            has_user_read = "User.Read" in target.split(" ")
        else:
            has_user_read = "User.Read" in target

        prev = home_to_best_at.get(home_id)
        if prev is None or expires_on > prev["expires_on"]:
            home_to_best_at[home_id] = {
                "secret": secret,
                "expires_on": expires_on,
                "client_id": tok.get("client_id"),
                "realm": tok.get("realm"),
                "scope": tok.get("target"),
                "has_user_read": has_user_read,
            }

    home_to_rt = {}
    for tok in refresh_tokens.values():
        home_id = tok.get("home_account_id")
        secret = tok.get("secret")
        if home_id and secret:
            home_to_rt[home_id] = {
                "secret": secret,
                "client_id": tok.get("client_id"),
            }

    for home_id, email in home_to_email.items():
        at = home_to_best_at.get(home_id, {})
        rt = home_to_rt.get(home_id, {})
        yield {
            "email": email,
            "access_token": at.get("secret"),
            "expires_on": at.get("expires_on", 0),
            "refresh_token": rt.get("secret"),
            "client_id": at.get("client_id") or rt.get("client_id"),
            "realm": at.get("realm"),
            "scope": at.get("scope"),
            "has_user_read": at.get("has_user_read", False),
        }


def main():
    report = {"checked_at": NOW.isoformat(), "accounts": []}

    # Google
    if GOOGLE_TOKEN_DIR.exists():
        for token_file in sorted(GOOGLE_TOKEN_DIR.glob("*.json")):
            email = token_file.stem
            if "@" not in email:
                continue  # skip non-account files like oauth_states.json
            age_days, age_st = age_status(token_file)
            status, detail = probe_google(token_file)
            if status == "unknown" and "deps missing" in detail:
                status = age_st  # fallback
                detail = f"probe deps unavailable, falling back to age ({age_days}d)"
            report["accounts"].append({
                "provider": "google_workspace",
                "email": email,
                "status": status,
                "detail": detail,
                "age_days": age_days,
                "file": token_file.name,
            })
    else:
        report["accounts"].append({
            "provider": "google_workspace",
            "note": "Token directory not found",
            "status": "not_configured",
        })

    # Outlook
    if OUTLOOK_CACHE.exists():
        try:
            entries = list(parse_outlook_cache(OUTLOOK_CACHE))
        except PermissionError as exc:
            entries = []
            report["accounts"].append({
                "provider": "outlook",
                "status": "warning",
                "detail": f"cache perms unsafe: {exc}",
            })
        except Exception as exc:
            entries = []
            report["accounts"].append({
                "provider": "outlook",
                "status": "unknown",
                "detail": f"cache parse failed: {exc}",
            })

        cache_age_days, cache_age_st = age_status(OUTLOOK_CACHE)
        import time
        now_epoch = int(time.time())
        for acc in entries:
            email = acc["email"]
            access_token = acc.get("access_token")
            refresh_token = acc.get("refresh_token")
            expires_on = acc.get("expires_on", 0)

            # If access_token is missing or expired, try refresh first
            if access_token is None or expires_on <= now_epoch + 60:
                if refresh_token and acc.get("client_id") and acc.get("realm") and acc.get("scope"):
                    fresh, refresh_detail = refresh_outlook_token(
                        refresh_token, acc["client_id"], acc["realm"], acc["scope"]
                    )
                    if fresh:
                        access_token = fresh
                        refresh_note = "refreshed → "
                    else:
                        report["accounts"].append({
                            "provider": "outlook",
                            "email": email,
                            "status": "expired",
                            "detail": f"refresh failed: {refresh_detail}",
                            "age_days": cache_age_days,
                        })
                        continue
                else:
                    report["accounts"].append({
                        "provider": "outlook",
                        "email": email,
                        "status": "expired",
                        "detail": "access_token expired and no refresh_token in cache",
                        "age_days": cache_age_days,
                    })
                    continue
            else:
                refresh_note = ""

            status, detail = probe_outlook_account(access_token, account_has_user_read=acc.get("has_user_read", False))
            if status == "unknown" and "deps missing" in detail:
                status = cache_age_st
                detail = f"probe deps unavailable, falling back to age ({cache_age_days}d)"
            report["accounts"].append({
                "provider": "outlook",
                "email": email,
                "status": status,
                "detail": f"{refresh_note}{detail}",
                "age_days": cache_age_days,
            })
    else:
        report["accounts"].append({
            "provider": "outlook",
            "note": "Token cache not found",
            "status": "not_configured",
        })

    # Overall
    statuses = [a["status"] for a in report["accounts"]]
    report["expired_count"] = statuses.count("expired")
    report["warnings_count"] = statuses.count("warning")
    report["not_configured_count"] = statuses.count("not_configured")
    report["overall"] = (
        "critical" if report["expired_count"]
        else "warning" if report["warnings_count"]
        else "healthy"
    )

    print(json.dumps(report, indent=2))
    sys.exit(0 if report["overall"] == "healthy" else 1)


if __name__ == "__main__":
    main()
