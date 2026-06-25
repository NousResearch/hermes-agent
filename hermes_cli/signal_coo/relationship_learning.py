"""Approval-gated relationship learning for Torben email triage."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .action_ledger import ActionLedger


RELATIONSHIP_LEARNING_VERSION = 1
ACTION_KEY_PREFIX = "relationship_learning"
OPEN_STATUSES = {"staged", "approval_required", "approved", "executing"}
DISCARD_TERMS = ("not important", "ignore", "discard", "spam", "noise", "do not surface", "dont surface")


def learned_contacts_path_for(context_path: str | Path) -> Path:
    return Path(context_path).expanduser().parent / "learned_contacts.yaml"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _utc_now()).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sender_name(candidate: dict[str, Any]) -> str:
    sender = str(candidate.get("sender") or "").strip()
    cleaned = re.sub(r"<[^<>]+>", "", sender).strip().strip('"')
    return cleaned or str(candidate.get("sender_email") or "Unknown contact")


def _action_key(candidate: dict[str, Any]) -> str:
    sender_email = str(candidate.get("sender_email") or "").lower()
    sender = str(candidate.get("sender") or "").lower()
    return f"{ACTION_KEY_PREFIX}:{sender_email or sender}"


def _existing_learning_actions(ledger: ActionLedger) -> dict[str, str]:
    existing: dict[str, str] = {}
    for record in ledger.load():
        state = record.executor_state or {}
        key = state.get("relationship_learning_key")
        if (
            record.scope == "ea"
            and isinstance(key, str)
            and record.status in OPEN_STATUSES
            and state.get("relationship_learning_version") == RELATIONSHIP_LEARNING_VERSION
        ):
            existing[key] = record.handle
    return existing


def stage_relationship_learning_actions(
    *,
    ledger: ActionLedger,
    candidates: list[dict[str, Any]],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    now = (now or _utc_now()).astimezone(timezone.utc)
    existing = _existing_learning_actions(ledger)
    staged: list[dict[str, Any]] = []
    for candidate in candidates:
        key = _action_key(candidate)
        handle = existing.get(key)
        name = _sender_name(candidate)
        question = str(candidate.get("question_for_eric") or f"Who is {name}, and when should I surface their emails?")
        if handle is None:
            record = ledger.add_action(
                scope="EA",
                summary=f"Learn contact: {name}",
                evidence_ids=[str(item) for item in (candidate.get("evidence_ids") or [])],
                allowed_next_actions=["answer_learn_contact", "discard"],
                status="approval_required",
                risk_class="low",
                ttl_hours=30 * 24,
                now=now,
                executor_state={
                    "mutation_type": "relationship_learning",
                    "provider": "local_config",
                    "mutation_status": "not_learned",
                    "relationship_learning_version": RELATIONSHIP_LEARNING_VERSION,
                    "relationship_learning_key": key,
                    "sender": candidate.get("sender"),
                    "sender_email": candidate.get("sender_email"),
                    "sender_domain": candidate.get("sender_domain"),
                    "subject": candidate.get("subject"),
                    "account": candidate.get("account") or candidate.get("account_alias"),
                    "observed_context": candidate.get("observed_context"),
                    "question_for_eric": question,
                    "source": candidate.get("source"),
                },
            )
            handle = record.handle
            existing[key] = handle
        staged.append({**candidate, "handle": handle, "question_for_eric": question})
    return staged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


def _answer_discards(answer: str) -> bool:
    normalized = f" {answer.strip().lower()} "
    return any(term in normalized for term in DISCARD_TERMS)


def _infer_role(answer: str) -> str:
    text = answer.lower()
    role_terms = [
        ("investor", "investor"),
        ("vc", "investor"),
        ("funding", "investor"),
        ("wife", "spouse"),
        ("spouse", "spouse"),
        ("teacher", "teacher"),
        ("school", "school_contact"),
        ("customer", "customer"),
        ("client", "customer"),
        ("founder", "founder_peer"),
        ("friend", "friend"),
        ("family", "family"),
        ("insurance", "insurance"),
        ("doctor", "medical"),
        ("recruiter", "recruiter"),
    ]
    for term, role in role_terms:
        if term in text:
            return role
    return "learned_contact"


def _infer_importance(answer: str, role: str) -> str:
    text = answer.lower()
    if any(term in text for term in ("critical", "always", "wife", "spouse", "kid", "son", "teacher")):
        return "critical"
    if role in {"investor", "customer", "school_contact", "medical"}:
        return "high"
    if any(term in text for term in ("important", "surface", "flag", "funding", "client")):
        return "high"
    if any(term in text for term in ("low", "not urgent", "only daily")):
        return "low"
    return "medium"


def _infer_surface_when(answer: str, role: str) -> list[str]:
    text = answer.lower()
    values = {"direct_ask", "scheduling", "follow_up"}
    if role == "investor" or "funding" in text:
        values.add("funding_context")
    if role in {"spouse", "family", "friend"} or "family" in text:
        values.add("family_admin")
    if role in {"teacher", "school_contact"} or "school" in text:
        values.add("childcare")
        values.add("school")
    if "only daily" in text or "daily brief" in text:
        values.add("daily_brief")
    if "realtime" in text or "immediately" in text:
        values.add("realtime")
    return sorted(values)


def _upsert_contact(payload: dict[str, Any], contact: dict[str, Any]) -> dict[str, Any]:
    people = list(payload.get("people") or [])
    email = str(contact.get("email") or "").lower()
    name = str(contact.get("name") or "").lower()
    updated = False
    for index, person in enumerate(people):
        aliases = [str(item).lower() for item in (person.get("aliases") or [])]
        emails = [str(item).lower() for item in (person.get("emails") or [])]
        if (email and (email in emails or email in aliases)) or (name and str(person.get("name") or "").lower() == name):
            merged = {**person, **contact}
            merged_aliases = list(dict.fromkeys(list(person.get("aliases") or []) + list(contact.get("aliases") or [])))
            merged["aliases"] = merged_aliases
            if email:
                merged["emails"] = list(dict.fromkeys(list(person.get("emails") or []) + [email]))
            people[index] = merged
            updated = True
            break
    if not updated:
        people.append(contact)
    return {**payload, "version": RELATIONSHIP_LEARNING_VERSION, "updated_at": _iso(), "people": people}


def apply_relationship_learning_answer(
    *,
    ledger: ActionLedger,
    relationship_context_path: str | Path,
    handle: str,
    answer: str,
    approved_by: str = "signal-reply",
    now: datetime | None = None,
) -> dict[str, Any]:
    now = (now or _utc_now()).astimezone(timezone.utc)
    record = ledger.get(handle)
    if record is None:
        raise ValueError(f"No action found for handle: {handle}")
    state = dict(record.executor_state or {})
    if state.get("mutation_type") != "relationship_learning":
        raise ValueError(f"Action {handle} is not a relationship-learning action.")
    if record.status not in OPEN_STATUSES:
        raise ValueError(f"Action {handle} is not open for relationship learning: {record.status}")
    answer = answer.strip()
    if not answer:
        raise ValueError("Relationship learning answer cannot be empty.")

    result: dict[str, Any] = {
        "handle": handle,
        "approved_by": approved_by,
        "external_mutations": 0,
        "local_config_mutations": 0,
        "discarded": False,
        "learned_contact": None,
        "learned_contacts_path": str(learned_contacts_path_for(relationship_context_path)),
    }
    records = ledger.load()
    if _answer_discards(answer):
        for candidate in records:
            if candidate.handle == handle:
                candidate.status = "discarded"
                candidate.executor_state.update(
                    {"mutation_status": "discarded", "learned_answer": answer, "applied_at": _iso(now)}
                )
                candidate.resolution_history.append(
                    {"at": _iso(now), "status": "discarded", "reason": f"Relationship learning discarded by {approved_by}."}
                )
        ledger.save(records)
        result["discarded"] = True
        return result

    name = _sender_name(state)
    sender_email = str(state.get("sender_email") or "").lower()
    role = _infer_role(answer)
    contact = {
        "name": name,
        "aliases": [item for item in [name, sender_email] if item],
        "emails": [sender_email] if sender_email else [],
        "role": role,
        "importance": _infer_importance(answer, role),
        "surface_when": _infer_surface_when(answer, role),
        "notes": f"Learned from Eric: {answer}",
        "learned_from": {
            "handle": handle,
            "at": _iso(now),
            "source": "signal_reply",
            "sender_domain": state.get("sender_domain"),
        },
    }
    learned_path = learned_contacts_path_for(relationship_context_path)
    payload = _upsert_contact(_load_yaml(learned_path), contact)
    _write_yaml(learned_path, payload)

    for candidate in records:
        if candidate.handle == handle:
            candidate.status = "executed"
            candidate.executor_state.update(
                {
                    "mutation_status": "learned",
                    "learned_answer": answer,
                    "learned_contact": contact,
                    "learned_contacts_path": str(learned_path),
                    "applied_at": _iso(now),
                }
            )
            candidate.resolution_history.append(
                {"at": _iso(now), "status": "learned", "reason": f"Relationship learned from {approved_by}."}
            )
    ledger.save(records)
    result["local_config_mutations"] = 1
    result["learned_contact"] = contact
    return result
