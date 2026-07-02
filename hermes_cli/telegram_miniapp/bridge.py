"""Action-gate bridge between the Telegram Mini App sidecar and the gateway.

The sidecar (a separate process) cannot call the gateway's in-memory approval
queue directly. This module is the narrow, file-backed, HMAC-authenticated
contract that lets an owner's ``approve_once`` / ``reject_once`` decision reach
``resolve_gateway_approval`` — and nothing else.

Design invariants (M18-alt Phase 1):
- The public snapshot never contains raw command, paths, env, PIDs, session
  keys, pattern keys, prompts, or tokens — only opaque ids and safe summaries.
- Opaque approval ids are HMAC-derived and reveal neither the session key nor
  the command.
- Every decision is HMAC-signed with a key derived from the bot token, so a
  third local process without the token cannot forge one.
- Resolution is fail-closed: bad signature, expired TTL, stale snapshot
  version, unknown id, disallowed decision, or a replayed request id all refuse
  without resolving anything. Only ``approve_once``/``reject_once`` are mapped,
  and never ``resolve_all``.

This module is pure and side-effect-free except for the file operations on the
``MiniAppBridge`` instance; it is not wired into the live gateway unless the
owner explicitly enables the bridge and restarts the gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

_DECISION_TTL_SECONDS = 120
_SNAPSHOT_FRESH_SECONDS = 30
_ALLOWED_DECISIONS = {"approve_once": "once", "reject_once": "deny"}
_ALLOWED_RISKS = {"read_only", "reversible", "risky", "critical"}


def derive_bridge_key(bot_token: str) -> bytes:
    """Derive the bridge HMAC key from the bot token.

    Both processes know the bot token; a local process that does not cannot
    forge a decision or predict an opaque id.
    """
    return hmac.new(b"hermes-miniapp-bridge", (bot_token or "").encode(), hashlib.sha256).digest()


def opaque_approval_id(bridge_key: bytes, *, session_key: str, command: str, seq: int = 0) -> str:
    # Stable identity from session + command + FIFO position only. It does NOT
    # depend on a timestamp: real gateway approvals carry no stable
    # ``requested_at``, so timing must not perturb the id (or the id/version
    # would churn every export tick and every decision would go stale). ``seq``
    # disambiguates two identical commands in the same session.
    material = f"{session_key}\x1f{command}\x1f{seq}".encode()
    return hmac.new(bridge_key, material, hashlib.sha256).hexdigest()[:16]


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sign_envelope(bridge_key: bytes, payload: dict[str, Any]) -> dict[str, Any]:
    sig = hmac.new(bridge_key, _canonical(payload), hashlib.sha256).hexdigest()
    return {"payload": payload, "sig": sig}


def verify_envelope(bridge_key: bytes, envelope: Any) -> dict[str, Any] | None:
    if not isinstance(envelope, dict):
        return None
    payload = envelope.get("payload")
    sig = envelope.get("sig")
    if not isinstance(payload, dict) or not isinstance(sig, str):
        return None
    expected = hmac.new(bridge_key, _canonical(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    return payload


@dataclass
class ApprovalRef:
    session_key: str
    is_head: bool  # True only for the oldest pending approval in its session


@dataclass
class ProjectedSnapshot:
    public: dict[str, Any]
    index: dict[str, ApprovalRef]  # approval_id -> ref (private, never serialised)


def _safe_risk(value: Any) -> str:
    return value if value in _ALLOWED_RISKS else "critical"


# The upstream approval ``description`` is free text that may embed the raw
# command, a path, or a secret. The bridge never copies it into the public
# snapshot; only a fixed, allowlisted summary keyed by risk tier is exposed.
_RISK_SUMMARY = {
    "read_only": "Безопасное чтение. Требуется решение владельца.",
    "reversible": "Обратимое действие. Требуется решение владельца.",
    "risky": "Рискованное действие. Требуется решение владельца.",
    "critical": "Опасное необратимое действие. Требуется решение владельца.",
}
_RISK_TITLE = {
    "read_only": "Чтение данных",
    "reversible": "Обратимое действие",
    "risky": "Рискованное действие",
    "critical": "Опасная команда",
}


def _coarse_time(value: Any) -> str:
    # Minute-resolution string; avoids leaking precise scheduling while staying
    # useful. Non-numeric values pass through as-is (already coarse labels).
    if isinstance(value, (int, float)):
        return str(int(value) // 60 * 60)
    return str(value) if value is not None else ""


def project_snapshot(bridge_key: bytes, pendings: Iterable[dict[str, Any]], *, now: int | float) -> ProjectedSnapshot:
    """Project raw pending approvals into a redacted public snapshot.

    ``pendings`` items carry sensitive fields (session_key, command, ...); the
    public output carries only opaque ids and safe summaries.
    """
    items: list[dict[str, Any]] = []
    index: dict[str, ApprovalRef] = {}
    seen_sessions: set[str] = set()
    for seq, pending in enumerate(pendings):
        session_key = str(pending.get("session_key", ""))
        command = str(pending.get("command", ""))
        approval_id = opaque_approval_id(bridge_key, session_key=session_key, command=command, seq=seq)
        # The gateway resolves a session FIFO, so only the oldest pending per
        # session is head-resolvable; anything behind it is reject-on-decision.
        is_head = session_key not in seen_sessions
        seen_sessions.add(session_key)
        index[approval_id] = ApprovalRef(session_key=session_key, is_head=is_head)
        risk = _safe_risk(pending.get("risk_tier"))
        items.append(
            {
                "approval_id": approval_id,
                "title": _RISK_TITLE[risk],
                "source_label": "gateway approval",
                "risk": risk,
                "summary": _RISK_SUMMARY[risk],
                "requested_at": _coarse_time(pending.get("requested_at")),
                "expires_at": _coarse_time(pending.get("expires_at")),
                "allowed_decisions": ["approve_once", "reject_once"],
            }
        )
    # Version is content-addressed on the STABLE identity + risk of each item
    # (opaque id already folds in session/command/seq), never on timestamps, so
    # it stays constant across ticks while the queue is unchanged — a decision
    # signed against it survives re-exports until the queue actually changes.
    stable = [{"approval_id": i["approval_id"], "risk": i["risk"]} for i in items]
    version = hashlib.sha256(_canonical({"items": stable})).hexdigest()[:16]
    body = {"generated_at": _coarse_time(now), "items": items, "snapshot_version": version}
    return ProjectedSnapshot(public=body, index=index)


@dataclass
class DecisionResult:
    accepted: bool
    status: str
    session_key: str | None = None
    choice: str | None = None
    decision_id: str | None = None


def _decision_id(payload: dict[str, Any]) -> str:
    # Include decision + snapshot_version so two distinct decisions that happen
    # to reuse a client_request_id do not collide into one file / one id.
    material = "\x1f".join(
        str(payload.get(field))
        for field in ("client_request_id", "approval_id", "decision", "snapshot_version")
    ).encode()
    return hashlib.sha256(material).hexdigest()[:16]


def validate_decision(
    bridge_key: bytes,
    envelope: Any,
    *,
    current_snapshot_version: str,
    index: dict[str, ApprovalRef],
    now: int | float,
    applied: set[str],
) -> DecisionResult:
    """Fail-closed validation of a signed decision envelope."""
    payload = verify_envelope(bridge_key, envelope)
    if payload is None:
        return DecisionResult(False, "rejected_invalid_signature")

    decision_id = _decision_id(payload)
    if decision_id in applied:
        return DecisionResult(False, "already_resolved", decision_id=decision_id)

    issued_at = payload.get("issued_at")
    if not isinstance(issued_at, (int, float)) or abs(now - issued_at) > _DECISION_TTL_SECONDS:
        return DecisionResult(False, "rejected_expired", decision_id=decision_id)

    if payload.get("snapshot_version") != current_snapshot_version:
        return DecisionResult(False, "rejected_stale", decision_id=decision_id)

    choice = _ALLOWED_DECISIONS.get(payload.get("decision"))
    if choice is None:
        return DecisionResult(False, "rejected_unsupported", decision_id=decision_id)

    ref = index.get(payload.get("approval_id"))
    if ref is None:
        return DecisionResult(False, "rejected_unknown", decision_id=decision_id)

    # The gateway resolver acts FIFO per session; only the head approval maps to
    # the item the owner actually selected. Refuse anything behind it so a
    # decision can never resolve a different (e.g. more dangerous) command.
    if not ref.is_head:
        return DecisionResult(False, "rejected_not_head", decision_id=decision_id)

    return DecisionResult(True, "accepted", session_key=ref.session_key, choice=choice, decision_id=decision_id)


class MiniAppBridge:
    """File-backed bridge rooted at ``<home>/miniapp`` (perms 0700/0600)."""

    def __init__(self, home: os.PathLike[str] | str, bridge_key: bytes) -> None:
        self._key = bridge_key
        self._root = Path(home) / "miniapp"
        self._decisions = self._root / "decisions"
        self._receipts = self._root / "receipts"
        self._snapshot = self._root / "approvals_snapshot.json"
        self._audit = self._root / "audit.jsonl"
        self._version = ""
        self._index: dict[str, str] = {}
        self._applied: set[str] = set()

    # ── directory / file helpers ─────────────────────────────────────────
    def _ensure_dirs(self) -> None:
        for path in (self._root, self._decisions, self._receipts):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                path.chmod(0o700)
            except OSError:
                pass

    def _atomic_write(self, path: Path, data: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
        os.replace(tmp, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass

    # ── gateway side: export snapshot ────────────────────────────────────
    def export(self, pendings: Iterable[dict[str, Any]], *, now: int | float) -> str:
        self._ensure_dirs()
        projected = project_snapshot(self._key, pendings, now=now)
        self._version = projected.public["snapshot_version"]
        self._index = projected.index
        # Sign the whole displayed body: a local process without the bot token
        # cannot downgrade the shown risk/title without invalidating the sig, so
        # the owner never sees tampered decision context.
        self._atomic_write(self._snapshot, json.dumps(sign_envelope(self._key, projected.public), ensure_ascii=False))
        return self._version

    # ── sidecar side: read snapshot, submit decision ─────────────────────
    def read_public_snapshot(self, *, now: int | float) -> dict[str, Any] | None:
        try:
            raw = self._snapshot.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            envelope = json.loads(raw)
        except (ValueError, TypeError):
            return None
        # Verify the exporter's signature: reject any locally tampered body.
        data = verify_envelope(self._key, envelope)
        if data is None:
            return None
        generated = data.get("generated_at")
        if isinstance(generated, str) and generated.isdigit():
            if now - int(generated) > _SNAPSHOT_FRESH_SECONDS + 60:
                return None
        return data

    def check_target(self, approval_id: str, snapshot_version: str, *, now: int | float) -> bool:
        """Sidecar-side pre-check before signing a decision.

        Rejects acting on a stale saved snapshot: the current on-disk snapshot
        must be fresh, its version must equal the client's, and the target
        approval_id must be present. The gateway still re-validates against its
        own live index — this is the earlier, defence-in-depth gate.
        """
        snapshot = self.read_public_snapshot(now=now)
        if snapshot is None:
            return False
        if str(snapshot.get("snapshot_version", "")) != snapshot_version:
            return False
        return any(
            isinstance(item, dict) and item.get("approval_id") == approval_id
            for item in snapshot.get("items", []) or []
        )

    def submit_decision(self, envelope: dict[str, Any]) -> str | None:
        """Persist a signed decision for the gateway to pick up.

        Basic shape/signature check only; full authorisation happens gateway
        side in :meth:`process_pending_decisions`.
        """
        payload = verify_envelope(self._key, envelope)
        if payload is None:
            return None
        self._ensure_dirs()
        decision_id = _decision_id(payload)
        self._atomic_write(self._decisions / f"{decision_id}.json", json.dumps(envelope, ensure_ascii=False))
        return decision_id

    def head_id_for_session(self, session_key: str) -> str | None:
        """Opaque id of the current FIFO-head approval for a session, from the
        most recent export. Used to re-bind decisions to the live head."""
        for approval_id, ref in self._index.items():
            if ref.session_key == session_key and ref.is_head:
                return approval_id
        return None

    # ── gateway side: process decisions ──────────────────────────────────
    def process_pending_decisions(
        self,
        resolver: Callable[..., int],
        *,
        now: int | float,
        current_head: Callable[[str], str | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve queued decisions against the gateway.

        ``resolver(session_key, choice, *, expected_approval_id)`` MUST perform
        an atomic compare-and-pop: under the gateway's approval lock, resolve the
        session's FIFO-head ONLY if it still equals ``expected_approval_id``,
        else resolve nothing and return 0. This closes the TOCTOU window — the
        head cannot change between the check and the pop, so a decision can
        never unblock a different (possibly more dangerous) command.

        ``expected_approval_id`` is passed KEYWORD-ONLY on purpose. The upstream
        ``tools.approval.resolve_gateway_approval(session_key, choice,
        resolve_all=False)`` takes a positional third arg meaning *resolve_all*;
        passing our id positionally there would make a single approve_once
        unblock the WHOLE session. Keyword-only makes that raw resolver raise
        TypeError instead — the loop catches it and fails closed. The live
        gateway wiring must supply a dedicated CAS adapter, never the raw
        resolver.

        ``current_head`` is an additional early, non-authoritative reject using
        this bridge's most-recent export index (defaults to it, so the check
        always runs); the resolver's atomic CAS is the real guarantee.
        """
        head_fn = current_head if current_head is not None else self.head_id_for_session
        self._ensure_dirs()
        receipts: list[dict[str, Any]] = []
        for path in sorted(self._decisions.glob("*.json")):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                self._safe_unlink(path)
                continue
            result = validate_decision(
                self._key,
                envelope,
                current_snapshot_version=self._version,
                index=self._index,
                now=now,
                applied=self._applied,
            )
            payload = envelope.get("payload") if isinstance(envelope, dict) else None
            approval_id = payload.get("approval_id") if isinstance(payload, dict) else None
            if result.accepted and result.session_key and result.choice:
                # Early, non-authoritative reject against the exported index.
                if head_fn(result.session_key) != approval_id:
                    result = DecisionResult(False, "rejected_not_head", decision_id=result.decision_id)
                else:
                    try:
                        # Atomic CAS: resolver resolves the head only if it still
                        # equals approval_id, else returns 0. Keyword-only so the
                        # raw resolve_gateway_approval (positional resolve_all)
                        # can never be mis-wired into a resolve_all.
                        count = resolver(result.session_key, result.choice, expected_approval_id=approval_id)
                    except Exception:  # noqa: BLE001 - a resolver failure must not crash the loop
                        result = DecisionResult(False, "rejected_resolver_error", decision_id=result.decision_id)
                    else:
                        if not count or count <= 0:
                            # Nothing was pending (timed out / resolved elsewhere)
                            # — report honestly rather than a false "accepted".
                            result = DecisionResult(False, "rejected_not_pending", decision_id=result.decision_id)
            if result.decision_id:
                self._applied.add(result.decision_id)
            receipt = {
                "decision_id": result.decision_id,
                "status": result.status,
                "risk_tier": "redacted",
                "resolved_at": _coarse_time(now),
                "redaction": "safe-receipt",
            }
            if result.decision_id:
                self._atomic_write(self._receipts / f"{result.decision_id}.json", json.dumps(receipt, ensure_ascii=False))
            self._append_audit(receipt)
            receipts.append(receipt)
            self._safe_unlink(path)
        return receipts

    # ── gateway side: one full driver cycle ──────────────────────────────
    def run_cycle(
        self, pendings: Iterable[dict[str, Any]], resolver: Callable[[str, str], int], *, now: int | float
    ) -> list[dict[str, Any]]:
        """One gateway tick: re-export the live queue (refreshing the head
        index), then process any queued decisions against that fresh state.

        This is the end-to-end primitive a gateway bridge thread calls each
        tick. It is deliberately synchronous and injectable so the full
        export -> sign -> process -> resolve loop is testable without a live
        gateway. Activating it on the running gateway (config bridge_enabled +
        restart) stays an explicit owner decision.
        """
        self.export(list(pendings), now=now)
        return self.process_pending_decisions(resolver, now=now)

    def _append_audit(self, receipt: dict[str, Any]) -> None:
        line = json.dumps(receipt, ensure_ascii=False) + "\n"
        fd = os.open(self._audit, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        try:
            self._audit.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass
