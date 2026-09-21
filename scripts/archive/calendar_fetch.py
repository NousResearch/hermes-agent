#!/usr/bin/env python3
"""Fetch calendar events via stdio MCP from Google Workspace and Outlook."""
import json, subprocess, os, sys, queue, threading, datetime as dt, re, time

TZ = os.environ.get("TZ", "Europe/London")

def parse_dt(s):
    if not s:
        return None
    # Handle Z suffix
    s = s.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        # Try date only
        try:
            return dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).astimezone()
        except Exception:
            return None

class MCPClient:
    def __init__(self, command, env=None):
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **(env or {})}
        )
        self.req_id = 0
        self.lock = threading.Lock()
        self.out_q = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        # initialize
        self._send({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"kensei-cal","version":"1.0"}}})
        self.out_q.get(timeout=30)

    def _reader(self):
        for line in self.proc.stdout:
            try:
                self.out_q.put(json.loads(line))
            except Exception:
                pass

    def _send(self, msg):
        raw = json.dumps(msg) + "\n"
        self.proc.stdin.write(raw)
        self.proc.stdin.flush()

    def call(self, method, params=None):
        with self.lock:
            self.req_id += 1
            rid = self.req_id
        self._send({"jsonrpc":"2.0","id":rid,"method":method,"params":params or {}})
        while True:
            msg = self.out_q.get(timeout=60)
            if msg.get("id") == rid:
                return msg

    def tool(self, name, arguments):
        resp = self.call("tools/call", {"name": name, "arguments": arguments})
        return resp.get("result", {})

    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

import yaml
with open("/home/kensei/.hermes/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
servers = cfg.get("mcp_servers", {})

all_events = []
issues = []

now_local = dt.datetime.now(dt.timezone.utc).astimezone()
today_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
week_end = today_start + dt.timedelta(days=8)
start_iso = today_start.isoformat()
end_iso = week_end.isoformat()

# ---------- Google Workspace ----------
if servers.get("google_workspace", {}).get("enabled"):
    gs = servers["google_workspace"]
    env = gs.get("env", {})
    cmd = [gs["command"], *gs.get("args", [])]
    cred_dir = "/home/kensei/.google_workspace_mcp/credentials"
    accounts = sorted([f.replace(".json","") for f in os.listdir(cred_dir) if f.endswith(".json") and f != "oauth_states.json"])
    if not accounts:
        issues.append("Google Workspace: no authenticated accounts")
    for acc in accounts:
        client = MCPClient(cmd, env=env)
        try:
            cals = client.tool("list_calendars", {"user_google_email": acc})
            cal_items = cals.get("content", [])
            calendars = []
            for item in cal_items:
                try:
                    data = json.loads(item.get("text", "[]"))
                    if isinstance(data, dict):
                        calendars.append(data)
                    elif isinstance(data, list):
                        calendars.extend(data)
                except Exception:
                    pass
            for cal in calendars:
                cid = cal.get("id", "primary")
                cname = cal.get("summary", "Calendar")
                try:
                    ev = client.tool("get_events", {
                        "user_google_email": acc,
                        "calendar_id": cid,
                        "start_date": start_iso,
                        "end_date": end_iso
                    })
                    for it in ev.get("content", []):
                        try:
                            events = json.loads(it.get("text", "[]"))
                        except Exception:
                            continue
                        if not isinstance(events, list):
                            events = [events] if isinstance(events, dict) else []
                        for e in events:
                            if isinstance(e, dict):
                                all_events.append({
                                    "source": "Google",
                                    "account": acc,
                                    "calendar": cname,
                                    "summary": e.get("summary", "Untitled"),
                                    "start": e.get("start", {}).get("dateTime") or e.get("start", {}).get("date"),
                                    "end": e.get("end", {}).get("dateTime") or e.get("end", {}).get("date"),
                                    "location": e.get("location", ""),
                                    "link": e.get("htmlLink", "")
                                })
                except Exception as e2:
                    issues.append(f"Google {acc} {cid}: {str(e2)[:60]}")
        except Exception as e1:
            issues.append(f"Google {acc}: {str(e1)[:80]}")
        finally:
            client.close()

# ---------- Outlook ----------
if servers.get("outlook", {}).get("enabled"):
    osrv = servers["outlook"]
    env = osrv.get("env", {})
    cmd = [osrv["command"], *osrv.get("args", [])]
    client = MCPClient(cmd, env=env)
    try:
        resp = client.tool("list-accounts", {})
        acc_text = ""
        for it in resp.get("content", []):
            acc_text += it.get("text", "")
        accounts = []
        for m in re.findall(r"([\w.-]+@[\w.-]+\.[a-zA-Z]{2,})", acc_text):
            if m not in accounts:
                accounts.append(m)
        if not accounts:
            issues.append("Outlook: no accounts found")
        for acc in accounts:
            try:
                cals_resp = client.tool("list-calendars", {"account": acc})
                cal_ids = []
                for it in cals_resp.get("content", []):
                    text = it.get("text", "")
                    for m in re.findall(r'"id"\s*:\s*"([^"]+)"', text):
                        if m not in cal_ids:
                            cal_ids.append(m)
                for cid in cal_ids:
                    try:
                        ev_resp = client.tool("get-calendar-view", {
                            "account": acc,
                            "start": start_iso,
                            "end": end_iso,
                            "limit": 50
                        })
                        for it in ev_resp.get("content", []):
                            try:
                                data = json.loads(it.get("text", "[]"))
                            except Exception:
                                continue
                            if isinstance(data, dict):
                                data = data.get("value", [])
                            if not isinstance(data, list):
                                continue
                            for e in data:
                                if isinstance(e, dict):
                                    loc = e.get("location", {})
                                    loc_str = loc.get("displayName", "") if isinstance(loc, dict) else str(loc)
                                    all_events.append({
                                        "source": "Outlook",
                                        "account": acc,
                                        "calendar": "Calendar",
                                        "summary": e.get("subject", "Untitled"),
                                        "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date", "")),
                                        "end": e.get("end", {}).get("dateTime", e.get("end", {}).get("date", "")),
                                        "location": loc_str,
                                        "link": e.get("webLink", "")
                                    })
                    except Exception as e2:
                        issues.append(f"Outlook {acc} cal={cid}: {str(e2)[:60]}")
            except Exception as e1:
                issues.append(f"Outlook {acc}: {str(e1)[:80]}")
    except Exception as e:
        issues.append(f"Outlook: {str(e)[:80]}")
    finally:
        client.close()

# Deduplicate
seen = set()
unique = []
for ev in all_events:
    key = (ev["summary"], ev["start"])
    if key not in seen:
        seen.add(key)
        unique.append(ev)
unique.sort(key=lambda x: x["start"] or "")

today = [e for e in unique if e["start"] and parse_dt(e["start"]).date() == now_local.date()]
week = [e for e in unique if e["start"] and parse_dt(e["start"]).date() in [now_local.date() + dt.timedelta(days=d+1) for d in range(7)]]

os.makedirs("/home/kensei/.hermes/runbooks/calendar-brief/2026-05-12", exist_ok=True)
with open("/home/kensei/.hermes/runbooks/calendar-brief/2026-05-12/events.json", "w", encoding="utf-8") as f:
    json.dump({"today": today, "week": week, "issues": issues, "today_count": len(today), "week_count": len(week)}, f, indent=2, default=str)

print(json.dumps({"today": len(today), "week": len(week), "issues": issues}, indent=2))
