"""Bounded self-evolution (Jarvis Layer G / Phase 6).

A reflection pass reviews the agent's own telemetry and memory and produces
*proposals* for improving itself. Proposals land in an approval inbox
(data/proposals.json). Per the configured policy only the safest, reversible
category — memory cleanup — applies automatically; everything else waits for a
human click. Every applied change snapshots the whole hub first (via the
server's backup system) so it is one-click reversible.

Nothing here edits code. Applyable changes touch data files only:
- memory_prune     → de-duplicates data/memory.md (AUTO)
- prompt_addendum  → appends a learned guideline to data/agent_notes.md, which
                     is injected into the agent's system prompt (APPROVAL)
Structural suggestions (routing, new tools) are recorded as advisory proposals
for the human to act on; they never self-apply.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone

# Only this kind may apply without a human click (configurable policy).
AUTO_APPLY = {"memory_prune"}
MAX_PROPOSALS = 100
_FACT_RE = re.compile(r"- (?:\(\d{4}-\d{2}-\d{2}\) )?(.*)")


def dedupe_memory(text: str) -> tuple[str, list[str]]:
    """Return (deduped_text, removed_lines) — drops repeat facts, keeps order."""
    seen: set[str] = set()
    out: list[str] = []
    removed: list[str] = []
    for line in text.splitlines():
        if line.startswith("- "):
            m = _FACT_RE.match(line)
            key = (m.group(1) if m else line).strip().lower()
            if key and key in seen:
                removed.append(line)
                continue
            seen.add(key)
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), removed


class Reflection:
    def __init__(self, path, api) -> None:
        self.path = path
        self.api = api
        self._lock = threading.Lock()
        self._data = {"proposals": [], "next_id": 1}
        if path.exists():
            try:
                self._data.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")

    def list_proposals(self) -> list[dict]:
        with self._lock:
            return json.loads(json.dumps(self._data["proposals"]))

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for p in self._data["proposals"] if p["status"] == "pending")

    def _add(self, kind: str, title: str, rationale: str, payload: dict) -> dict:
        prop = {
            "id": self._data["next_id"],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "title": title,
            "rationale": rationale,
            "payload": payload,
            "status": "pending",
        }
        self._data["next_id"] += 1
        self._data["proposals"].append(prop)
        self._data["proposals"] = self._data["proposals"][-MAX_PROPOSALS:]
        return prop

    def _has_open(self, kind: str, title: str) -> bool:
        return any(p["kind"] == kind and p["title"] == title and p["status"] == "pending"
                   for p in self._data["proposals"])

    # -- reflection ----------------------------------------------------------
    def reflect(self) -> list[dict]:
        """Analyze telemetry + memory, queue proposals, auto-apply safe ones.
        Returns the proposals created this pass."""
        created: list[dict] = []
        with self._lock:
            for prop in self._observe():
                if self._has_open(prop["kind"], prop["title"]):
                    continue  # don't stack duplicates of a still-open proposal
                stored = self._add(prop["kind"], prop["title"], prop["rationale"], prop["payload"])
                if stored["kind"] in AUTO_APPLY:
                    self._apply_locked(stored)
                created.append(stored)
            self._save()
        return json.loads(json.dumps(created))

    def _observe(self) -> list[dict]:
        """Heuristic findings from real telemetry + memory (deterministic)."""
        findings: list[dict] = []

        # 1) duplicate facts in long-term memory → prune (auto)
        memory = self.api.memory_read() if self.api else ""
        _, removed = dedupe_memory(memory)
        if removed:
            findings.append({
                "kind": "memory_prune",
                "title": "De-duplicate long-term memory",
                "rationale": f"Found {len(removed)} repeated fact(s) in memory.",
                "payload": {"removed": removed[:20], "count": len(removed)},
            })

        # 2) tools the user keeps declining → propose a standing guideline
        events = self.api.telemetry.recent(500) if self.api else []
        denials: dict[str, int] = {}
        for e in events:
            if e.get("kind") == "tool" and e.get("approved") is False:
                denials[e.get("name", "?")] = denials.get(e.get("name", "?"), 0) + 1
        for name, n in denials.items():
            if n >= 2:
                findings.append({
                    "kind": "prompt_addendum",
                    "title": f"Stop proposing {name}",
                    "rationale": f"You declined “{name}” {n} times — the agent should stop offering it unprompted.",
                    "payload": {"text": f"The user has repeatedly declined the “{name}” action; do not use it unless they explicitly ask."},
                })

        # 3) frequent escalations → propose a decisiveness guideline
        summary = self.api.telemetry.summary() if self.api else {}
        if summary.get("escalations", 0) >= 3:
            findings.append({
                "kind": "prompt_addendum",
                "title": "Be more decisive",
                "rationale": f"The agent escalated {summary['escalations']} times; a clearer default would cut cost.",
                "payload": {"text": "When uncertain, state your assumptions and give a best-effort answer rather than hedging."},
            })

        # 4) model-augmented reflection (claude mode only): the deep tier may
        # suggest richer prompt_addendum guidelines. These are advisory and, per
        # policy, still require a human click — they never auto-apply. Deduped by
        # title against the heuristic findings above.
        assistant = getattr(self.api, "assistant", None)
        if assistant is not None and hasattr(assistant, "reflect_candidates"):
            seen_titles = {f["title"].lower() for f in findings}
            context = {
                "telemetry": summary,
                "recent_tools": [e for e in (self.api.telemetry.recent(30) if self.api else [])
                                 if e.get("kind") == "tool"][:20],
                "memory": memory[-2000:],
                "current_guidelines": (self.api.agent_notes_read() if self.api else "")[-1500:],
            }
            try:
                for cand in assistant.reflect_candidates(context):
                    if cand["kind"] == "prompt_addendum" and cand["title"].lower() not in seen_titles:
                        seen_titles.add(cand["title"].lower())
                        findings.append(cand)
            except Exception:
                pass  # model reflection is best-effort; never breaks a pass
        return findings

    # -- apply / dismiss -----------------------------------------------------
    def apply(self, proposal_id: int) -> dict:
        with self._lock:
            prop = self._find(proposal_id)
            if prop is None:
                raise KeyError("no such proposal")
            if prop["status"] != "pending":
                raise ValueError(f"proposal is already {prop['status']}")
            self._apply_locked(prop)
            self._save()
            return json.loads(json.dumps(prop))

    def _apply_locked(self, prop: dict) -> None:
        # Snapshot first so any applied change is one-click reversible.
        snapshot = None
        try:
            snapshot = self.api.backup_now({}).get("name")
        except Exception:
            pass
        if prop["kind"] == "memory_prune":
            deduped, _ = dedupe_memory(self.api.memory_read())
            self.api.memory_overwrite(deduped)
        elif prop["kind"] == "prompt_addendum":
            self.api.agent_notes_append(prop["payload"].get("text", ""))
        prop["status"] = "auto-applied" if prop["kind"] in AUTO_APPLY else "applied"
        prop["applied_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if snapshot:
            prop["snapshot"] = snapshot

    def dismiss(self, proposal_id: int) -> dict:
        with self._lock:
            prop = self._find(proposal_id)
            if prop is None:
                raise KeyError("no such proposal")
            prop["status"] = "dismissed"
            self._save()
            return json.loads(json.dumps(prop))

    def rollback(self, proposal_id: int) -> dict:
        """Undo an applied proposal by restoring its pre-apply snapshot.

        Every apply snapshots the whole hub first; rollback replays that
        snapshot so a learned guideline or memory prune can be reverted with a
        single click. Only applied proposals that captured a snapshot qualify.
        """
        with self._lock:
            prop = self._find(proposal_id)
            if prop is None:
                raise KeyError("no such proposal")
            if prop["status"] not in ("applied", "auto-applied"):
                raise ValueError(f"proposal is {prop['status']}, not applied")
            snapshot = prop.get("snapshot")
            if not snapshot:
                raise ValueError("no snapshot was captured for this proposal")
            try:
                self.api.backup_restore({"name": snapshot})
            except Exception as exc:  # ApiError etc. → surface as a value error
                raise ValueError(f"rollback failed: {exc}") from None
            prop["status"] = "rolled-back"
            prop["rolled_back_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._save()
            return json.loads(json.dumps(prop))

    def history(self, limit: int = 30) -> list[dict]:
        """Applied/dismissed/rolled-back proposals, newest first (the audit view)."""
        with self._lock:
            done = [p for p in self._data["proposals"]
                    if p["status"] not in ("pending",)]
        return json.loads(json.dumps(done[::-1][:limit]))

    def _find(self, proposal_id: int) -> dict | None:
        return next((p for p in self._data["proposals"] if p["id"] == proposal_id), None)
