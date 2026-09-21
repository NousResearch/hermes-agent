#!/usr/bin/env python3
"""Direct API fetch using stored credentials."""
import json, os, sys, datetime as dt, time, requests

TZ = dt.timezone(dt.timedelta(hours=1))
now = dt.datetime.now(TZ)
today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
week_end = today_start + dt.timedelta(days=8)

all_events = []
issues = []

# ---------- Google Calendar ----------
try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    cred_path = "/home/kensei/.google_workspace_mcp/credentials/saghir.sahil@gmail.com.json"
    with open(cred_path, encoding="utf-8") as f:
        data = json.load(f)
    creds = Credentials(
        token=data["token"],
        refresh_token=data.get("refresh_token"),
        token_uri=data["token_uri"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        scopes=data.get("scopes", [])
    )
    if creds.expired and creds.refresh_token:
        import google.auth.transport.requests
        creds.refresh(google.auth.transport.requests.Request())
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    calendars = service.calendarList().list().execute().get("items", [])
    for cal in calendars:
        cid = cal["id"]
        cname = cal.get("summary", "Calendar")
        # skip birthday/holiday only if no real events likely
        events_result = service.events().list(
            calendarId=cid,
            timeMin=today_start.isoformat(),
            timeMax=week_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=50
        ).execute()
        for e in events_result.get("items", []):
            start = e.get("start", {}).get("dateTime") or e.get("start", {}).get("date")
            end = e.get("end", {}).get("dateTime") or e.get("end", {}).get("date")
            all_events.append({
                "source": "Google",
                "account": "saghir.sahil@gmail.com",
                "calendar": cname,
                "summary": e.get("summary", "Untitled"),
                "start": start,
                "end": end,
                "location": e.get("location", ""),
                "link": e.get("htmlLink", ""),
                "description": e.get("description", "")[:200]
            })
except Exception as e:
    issues.append(f"Google Calendar: {str(e)[:120]}")

# ---------- Outlook via Graph API ----------
try:
    cache_path = "/home/kensei/.config/ms-365-mcp-server/token-cache.json"
    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)
    inner = json.loads(data["data"])
    # Extract accounts
    accounts = []
    for k, v in inner["Account"].items():
        accounts.append(v)
    for acc in accounts:
        username = acc["username"]
        # Find access token
        access_token = None
        access_keys = [k for k in inner if "AccessToken" in k and acc["home_account_id"] in k]
        if access_keys:
            at = inner[access_keys[0]]
            access_token = at.get("secret", "")
        if not access_token:
            issues.append(f"Outlook {username}: no access token")
            continue
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json", "Prefer": 'outlook.timezone="Europe/London"'}
        today_iso = today_start.strftime("%Y-%m-%dT%H:%M:%S")
        week_iso = week_end.strftime("%Y-%m-%dT%H:%M:%S")
        url = f"https://graph.microsoft.com/v1.0/me/calendarview?startDateTime={today_iso}&endDateTime={week_iso}&$top=50"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 401:
            issues.append(f"Outlook {username}: token expired (401)")
            continue
        if r.status_code != 200:
            issues.append(f"Outlook {username}: HTTP {r.status_code}")
            continue
        events = r.json().get("value", [])
        for e in events:
            loc_obj = e.get("location", {})
            loc_str = loc_obj.get("displayName", "") if isinstance(loc_obj, dict) else str(loc_obj)
            all_events.append({
                "source": "Outlook",
                "account": username,
                "calendar": "Calendar",
                "summary": e.get("subject", "Untitled"),
                "start": e.get("start", {}).get("dateTime", ""),
                "end": e.get("end", {}).get("dateTime", ""),
                "location": loc_str,
                "link": e.get("webLink", ""),
                "description": e.get("bodyPreview", "")[:200]
            })
except Exception as e:
    issues.append(f"Outlook Graph: {str(e)[:120]}")

# Deduplicate and sort
def parse_dt(s):
    if not s:
        return dt.datetime.min.replace(tzinfo=TZ)
    s = s.replace("Z", "+00:00")
    try:
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=TZ)
        return d
    except Exception:
        try:
            d = dt.datetime.strptime(s, "%Y-%m-%d")
            return d.replace(tzinfo=TZ)
        except Exception:
            return dt.datetime.min.replace(tzinfo=TZ)

seen = set()
unique = []
for ev in all_events:
    key = (ev["summary"], ev["start"])
    if key not in seen:
        seen.add(key)
        unique.append(ev)
unique.sort(key=lambda x: parse_dt(x["start"]))

today_events = [e for e in unique if parse_dt(e["start"]).date() == now.date()]
week_events = [e for e in unique if parse_dt(e["start"]).date() > now.date() and parse_dt(e["start"]).date() <= (now + dt.timedelta(days=7)).date()]

os.makedirs("/home/kensei/.hermes/runbooks/calendar-brief/2026-05-12", exist_ok=True)
with open("/home/kensei/.hermes/runbooks/calendar-brief/2026-05-12/events.json", "w", encoding="utf-8") as f:
    json.dump({"today": today_events, "week": week_events, "issues": issues, "today_count": len(today_events), "week_count": len(week_events)}, f, indent=2, default=str)

print(json.dumps({"today": len(today_events), "week": len(week_events), "issues": issues}, indent=2))
