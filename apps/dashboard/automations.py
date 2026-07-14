"""Automations engine for Hermes Hub — the "recognize and automate" layer.

Rules live server-side (data/automations.json) so they run even when no
browser is open. A background thread evaluates them every TICK seconds and
appends fired notifications to a persistent ring buffer that clients poll.

Trigger types
  daily      {"type": "daily", "time": "08:00"}            — local server time
  market     {"type": "market", "symbol": "BTC", "percent": 5}
             fires when |24h change| crosses the threshold (edge-triggered)
  worldstate {"type": "worldstate", "level": "elevated"}
             fires when the overall index reaches the level or worse

Action types
  notify     {"type": "notify", "message": "..."}
  briefing   {"type": "briefing"} — generates a briefing from the synced
             dashboard state + current headlines and sends it as the body
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

TICK_SECONDS = 20
MAX_NOTIFICATIONS = 100
MAX_RULES = 50

LEVEL_RANK = {"stable": 0, "watch": 1, "elevated": 2, "critical": 3}
VALID_TRIGGERS = ("daily", "market", "worldstate")
VALID_ACTIONS = ("notify", "briefing")


def validate_rule(rule: dict) -> str | None:
    """Returns an error message, or None if the rule is well-formed."""
    trigger = rule.get("trigger") or {}
    action = rule.get("action") or {}
    kind = trigger.get("type")
    if kind not in VALID_TRIGGERS:
        return f"trigger.type must be one of {VALID_TRIGGERS}"
    if kind == "daily":
        import re
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(trigger.get("time", ""))):
            return "daily trigger needs time as HH:MM (24h)"
    if kind == "market":
        if not trigger.get("symbol"):
            return "market trigger needs a symbol (e.g. BTC)"
        try:
            if not 0 < float(trigger.get("percent", 0)) <= 100:
                return "market trigger percent must be in (0, 100]"
        except (TypeError, ValueError):
            return "market trigger percent must be a number"
    if kind == "worldstate" and trigger.get("level") not in ("watch", "elevated", "critical"):
        return "worldstate trigger level must be watch, elevated or critical"
    if action.get("type") not in VALID_ACTIONS:
        return f"action.type must be one of {VALID_ACTIONS}"
    if action.get("type") == "notify" and not str(action.get("message", "")).strip():
        return "notify action needs a message"
    if not str(rule.get("name", "")).strip():
        return "rule needs a name"
    return None


class Automations:
    """Rule store + evaluator. `api` supplies markets/worldstate/briefing."""

    def __init__(self, path: Path, api) -> None:
        self.path = path
        self.api = api
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._data = {"rules": [], "notifications": [], "next_rule": 1, "next_notif": 1}
        if path.exists():
            try:
                self._data.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass  # corrupted file: start fresh, will be rewritten on save

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")

    # -- rule CRUD -----------------------------------------------------------
    def list_rules(self) -> list[dict]:
        with self._lock:
            return json.loads(json.dumps(self._data["rules"]))

    def create_rule(self, rule: dict) -> dict:
        error = validate_rule(rule)
        if error:
            raise ValueError(error)
        with self._lock:
            if len(self._data["rules"]) >= MAX_RULES:
                raise ValueError(f"rule limit ({MAX_RULES}) reached")
            stored = {
                "id": self._data["next_rule"],
                "name": str(rule["name"]).strip()[:80],
                "enabled": True,
                "trigger": rule["trigger"],
                "action": rule["action"],
                "state": {},  # evaluator bookkeeping (last fire date, armed flags)
            }
            if rule["trigger"]["type"] == "daily":
                # don't fire retroactively on creation day
                now = datetime.now()
                if now.strftime("%H:%M") >= rule["trigger"]["time"]:
                    stored["state"]["last_date"] = now.strftime("%Y-%m-%d")
            self._data["next_rule"] += 1
            self._data["rules"].append(stored)
            self._save()
            return json.loads(json.dumps(stored))

    def delete_rule(self, rule_id: int) -> bool:
        with self._lock:
            before = len(self._data["rules"])
            self._data["rules"] = [r for r in self._data["rules"] if r["id"] != rule_id]
            changed = len(self._data["rules"]) != before
            if changed:
                self._save()
            return changed

    def toggle_rule(self, rule_id: int) -> dict | None:
        with self._lock:
            for rule in self._data["rules"]:
                if rule["id"] == rule_id:
                    rule["enabled"] = not rule["enabled"]
                    self._save()
                    return json.loads(json.dumps(rule))
        return None

    # -- notifications ---------------------------------------------------------
    def notifications_after(self, after: int) -> dict:
        with self._lock:
            items = [n for n in self._data["notifications"] if n["id"] > after]
            last = self._data["next_notif"] - 1
        return {"notifications": items, "last": last}

    def _notify(self, title: str, body: str, rule_id: int | None = None) -> None:
        entry = {
            "id": self._data["next_notif"],
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "title": title[:120],
            "body": body[:2000],
            "rule_id": rule_id,
        }
        self._data["next_notif"] += 1
        self._data["notifications"].append(entry)
        self._data["notifications"] = self._data["notifications"][-MAX_NOTIFICATIONS:]

    # -- evaluation ---------------------------------------------------------------
    def tick(self, now: datetime | None = None) -> int:
        """Evaluate all rules once. Returns how many fired. Thread-safe."""
        now = now or datetime.now()
        fired = 0
        with self._lock:
            for rule in self._data["rules"]:
                if not rule.get("enabled"):
                    continue
                try:
                    if self._evaluate(rule, now):
                        fired += 1
                except Exception:
                    continue  # a broken upstream must not kill the loop
            if fired:
                self._save()
        return fired

    def _evaluate(self, rule: dict, now: datetime) -> bool:
        trigger = rule["trigger"]
        state = rule.setdefault("state", {})

        if trigger["type"] == "daily":
            today = now.strftime("%Y-%m-%d")
            if state.get("last_date") == today:
                return False
            if now.strftime("%H:%M") < trigger["time"]:
                return False
            state["last_date"] = today
            self._fire(rule)
            return True

        if trigger["type"] == "market":
            data = self.api.markets({})
            symbol = trigger["symbol"].upper()
            asset = next((a for a in data["assets"] if a["symbol"].upper() == symbol), None)
            if asset is None:
                return False
            beyond = abs(asset["change24h"]) >= float(trigger["percent"])
            was_beyond = state.get("beyond", False)
            state["beyond"] = beyond
            if beyond and not was_beyond:  # edge-triggered: fire on crossing
                direction = "up" if asset["change24h"] >= 0 else "down"
                self._fire(rule, extra=(
                    f"{asset['name']} ({symbol}) is {direction} "
                    f"{abs(asset['change24h']):.2f}% in 24h — ${asset['price']:,}"))
                return True
            return False

        if trigger["type"] == "worldstate":
            data = self.api.worldstate({})
            level = data["overall"]["level"]
            reached = LEVEL_RANK[level] >= LEVEL_RANK[trigger["level"]]
            was_reached = state.get("reached", False)
            state["reached"] = reached
            if reached and not was_reached:
                hot = ", ".join(
                    d["name"] for d in data["domains"]
                    if LEVEL_RANK[d["level"]] >= LEVEL_RANK["elevated"]) or "—"
                self._fire(rule, extra=(
                    f"Global index {data['overall']['score']} ({level.upper()}). "
                    f"Domains to watch: {hot}."))
                return True
            return False

        return False

    def _fire(self, rule: dict, extra: str = "") -> None:
        action = rule["action"]
        if action["type"] == "briefing":
            body = self._build_briefing()
            if extra:
                body = extra + "\n\n" + body
            self._notify(f"⏰ {rule['name']}", body, rule["id"])
        else:
            body = action.get("message", "")
            if extra:
                body = (body + "\n" + extra).strip()
            self._notify(f"🔔 {rule['name']}", body, rule["id"])

    def _build_briefing(self) -> str:
        context = {}
        try:
            synced = self.api.state_store.get()
            state = synced.get("state") or {}
            context["tasks"] = [
                {"name": l.get("name"), "items": [
                    {"text": i.get("text"), "done": i.get("done")} for i in l.get("items", [])]}
                for l in state.get("tasks", {}).get("lists", [])
            ]
            context["events"] = state.get("calendar", {}).get("events", [])
        except Exception:
            pass
        try:
            context["headlines"] = [
                i["title"] for i in self.api.news({"topic": ["top"], "limit": ["6"]})["items"]]
            world = self.api.worldstate({})
            context["worldstate"] = {
                "score": world["overall"]["score"], "level": world["overall"]["level"]}
        except Exception:
            pass
        result = self.api.assistant.briefing({"context": context})
        return result["briefing"]

    # -- background loop ------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="automations")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(TICK_SECONDS):
            try:
                self.tick()
            except Exception:
                pass  # never die
