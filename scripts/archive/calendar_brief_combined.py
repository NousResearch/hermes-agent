#!/usr/bin/env python3
"""Combined calendar fetch: Google (direct API) + Outlook (refresh then direct API)."""
import json, os, sys, datetime as dt, re, time, requests as rq
from pathlib import Path

TZ = dt.timezone(dt.timedelta(hours=1))
now = dt.datetime.now(TZ)
today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
week_end = today_start + dt.timedelta(days=8)
start_iso = today_start.isoformat()
end_iso = week_end.isoformat()

all_events = []
issues = []


def _parse_dt(s):
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        try:
            return dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).astimezone()
        except Exception:
            return None

# ================== GOOGLE ==================
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    GOOGLE_DEPS = True
except ImportError:
    GOOGLE_DEPS = False
    issues.append("Google: deps missing (google-auth)")

if GOOGLE_DEPS:
    cred_dir = Path("/home/kensei/.google_workspace_mcp/credentials")
    for token_file in sorted(cred_dir.glob("*.json")):
        email = token_file.stem
        if "@" not in email:
            continue
        try:
            creds = Credentials.from_authorized_user_file(str(token_file))
            creds.refresh(Request())
            headers = {"Authorization": f"Bearer {creds.token}"}
            # List calendars
            r = rq.get("https://www.googleapis.com/calendar/v3/users/me/calendarList", headers=headers, timeout=30)
            r.raise_for_status()
            cals = r.json().get("items", [])
            for cal in cals:
                cid = cal.get("id", "primary")
                cname = cal.get("summary", "Calendar")
                # Skip unwanted calendars
                if cal.get("accessRole") == "freeBusyReader" and not cal.get("selected", False):
                    continue
                params = {
                    "calendarId": cid,
                    "timeMin": today_start.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "timeMax": week_end.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "singleEvents": "true",
                    "orderBy": "startTime",
                    "maxResults": "100"
                }
                r2 = rq.get(f"https://www.googleapis.com/calendar/v3/calendars/{cid}/events", headers=headers, params=params, timeout=30)
                if r2.status_code == 403:
                    continue
                r2.raise_for_status()
                for e in r2.json().get("items", []):
                    start = e.get("start", {}).get("dateTime") or e.get("start", {}).get("date", "")
                    end = e.get("end", {}).get("dateTime") or e.get("end", {}).get("date", "")
                    all_events.append({
                        "source": "Google",
                        "account": email,
                        "calendar": cname,
                        "summary": e.get("summary", "Untitled"),
                        "start": start,
                        "end": end,
                        "location": e.get("location", ""),
                        "link": e.get("htmlLink", "")
                    })
        except Exception as e:
            issues.append(f"Google {email}: {str(e)[:80]}")

# ================== OUTLOOK ==================
OUTLOOK_CACHE = Path("/home/kensei/.config/ms-365-mcp-server/token-cache.json")
if OUTLOOK_CACHE.exists():
    try:
        top = json.loads(OUTLOOK_CACHE.read_text(encoding="utf-8"))
        inner = json.loads(top["data"])
    except Exception as e:
        issues.append(f"Outlook: cache parse failed: {e}")
        inner = None

    if inner:
        accounts = {}
        for k, v in inner.get("Account", {}).items():
            accounts[v.get("home_account_id")] = v.get("username")

        # Build best access token per account
        home_to_best = {}
        for k, v in inner.get("AccessToken", {}).items():
            haid = v.get("home_account_id")
            if not haid:
                continue
            expires_on = int(v.get("expires_on", 0) or 0)
            prev = home_to_best.get(haid)
            if prev is None or expires_on > prev["expires_on"]:
                home_to_best[haid] = {
                    "secret": v.get("secret"),
                    "expires_on": expires_on,
                    "client_id": v.get("client_id"),
                    "realm": v.get("realm"),
                    "scope": v.get("target"),
                }

        # Refresh tokens
        home_to_rt = {}
        for k, v in inner.get("RefreshToken", {}).items():
            haid = v.get("home_account_id")
            if haid and v.get("secret"):
                home_to_rt[haid] = {
                    "secret": v.get("secret"),
                    "client_id": v.get("client_id"),
                }

        now_epoch = int(time.time())
        for haid, email in accounts.items():
            at = home_to_best.get(haid, {})
            rt = home_to_rt.get(haid, {})
            access_token = at.get("secret")
            expires_on = at.get("expires_on", 0)

            # Refresh if needed
            if not access_token or expires_on <= now_epoch + 60:
                refresh_token = rt.get("secret")
                client_id = at.get("client_id") or rt.get("client_id")
                realm = at.get("realm")
                scope = at.get("scope")
                if refresh_token and client_id and realm and scope:
                    url = f"https://login.microsoftonline.com/{realm}/oauth2/v2.0/token"
                    rr = rq.post(url, data={
                        "client_id": client_id,
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token",
                        "scope": scope,
                    }, timeout=10)
                    if rr.status_code == 200:
                        access_token = rr.json().get("access_token")
                    else:
                        issues.append(f"Outlook {email}: refresh failed ({rr.status_code})")
                        continue
                else:
                    issues.append(f"Outlook {email}: no refresh_token available")
                    continue

            if not access_token:
                issues.append(f"Outlook {email}: no access_token")
                continue

            try:
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Prefer": 'outlook.timezone="Europe/London"'
                }
                params = {
                    "startDateTime": start_iso,
                    "endDateTime": end_iso,
                    "$select": "subject,start,end,location,bodyPreview,webLink",
                    "$top": "50"
                }
                url = "https://graph.microsoft.com/v1.0/me/calendarview"
                r = rq.get(url, headers=headers, params=params, timeout=30)
                if r.status_code == 401:
                    issues.append(f"Outlook {email}: expired (401)")
                    continue
                r.raise_for_status()
                data = r.json()
                for e in data.get("value", []):
                    loc_obj = e.get("location", {})
                    loc_str = loc_obj.get("displayName", "") if isinstance(loc_obj, dict) else str(loc_obj)
                    all_events.append({
                        "source": "Outlook",
                        "account": email,
                        "calendar": "Calendar",
                        "summary": e.get("subject", "Untitled"),
                        "start": e.get("start", {}).get("dateTime", ""),
                        "end": e.get("end", {}).get("dateTime", ""),
                        "location": loc_str,
                        "link": e.get("webLink", "")
                    })
            except Exception as e2:
                issues.append(f"Outlook {email}: {str(e2)[:80]}")
else:
    issues.append("Outlook: token cache not found")

# Deduplicate
seen = set()
unique = []
for ev in all_events:
    key = (ev["summary"], ev["start"])
    if key not in seen:
        seen.add(key)
        unique.append(ev)

unique.sort(key=lambda x: x["start"] or "")

today_events = [e for e in unique if e["start"] and _parse_dt(e["start"]).date() == now.date()]
week_events = [e for e in unique if e["start"] and _parse_dt(e["start"]).date() in [now.date() + dt.timedelta(days=d+1) for d in range(7)]]

# Save JSON
out_dir = Path("/home/kensei/.hermes/runbooks/calendar-brief")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "events.json", "w", encoding="utf-8") as f:
    json.dump({"today": today_events, "week": week_events, "issues": issues, "today_count": len(today_events), "week_count": len(week_events)}, f, indent=2, default=str)

print(json.dumps({"today": len(today_events), "week": len(week_events), "issues": issues}, indent=2))
