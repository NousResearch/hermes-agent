"""Continuing a Group Chat on another computer after its host is lost: proofs, receipts and records.

A group moves to another computer in one of four ways (``website/docs/developer-guide/
group-chat-host-loss.md``), each with its own proof kind:

- ``attested``: the owner continues the group on one eligible computer by hand;
- ``certified``: in majority mode a standby takes over by itself once a majority of the voters
  promised it, each only after its lease to the old host ran out;
- ``evidence``: with exactly two voters, the standby takes over after the careful silence and
  signs what it observed;
- ``handover``: the host itself signs its exact history over (a stop, Desktop's sleep hook, or the
  owner's move).

Eligible means ``successor: true`` in the room's custody configuration: one of the owner's own
computers, or a member the owner designated whose own operator consented. Every move fences the
host's epoch at every computer it reaches. Each answers with a signed fence receipt
(``fence_and_promise`` in #105079: one successor per epoch, never revoked), its watermark, its
evidence of the room's runs and, where its operator pre-consented, fresh grants for its members.
The successor adopts the most complete copy and writes one ``authority.transition`` with the proof
and the receipts. Every other computer verifies that proof itself before its copy follows.

This module holds the formats, their verification against the room's own configuration, and the
durable records: per-room succession state, the lineage an installation learned, and
continuation consent. Custody, room identity keys and the verified mark belong to #104601 and
#99107 and are reached through the adapters below.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import compact_json, display_label, table_exists

FENCE_REQUEST = b"hermes.group.succession.fence-request.v1"
FENCE_RECEIPT = b"hermes.group.succession.fence-receipt.v1"
ATTESTATION = b"hermes.group.succession.attestation.v1"
LEARN = b"hermes.group.succession.learn.v1"
QUERY = b"hermes.group.succession.query.v1"
ANSWER = b"hermes.group.succession.answer.v1"
DECISION = b"hermes.group.succession.keep-decision.v1"
REPORT = b"hermes.group.succession.report.v1"
_SEAL_INFO = b"hermes.group.succession.seal.v1"

ATTESTATION_TEXT = "The group owner continued this group on this computer while its host could not be reached."
KEEP_TEXT = "The group owner kept this computer as the group's host after it was continued on two computers."
ANYWAY_TEXT = "The group owner continued this group on its host anyway while the host could not reach its voters."
RECOVER_TEXT = ("The computer this group's next step was promised to never took it, so its host continued the group "
                "at a fresh step.")
CERTIFICATE = b"hermes.group.succession.certificate.v1"
EVIDENCE = b"hermes.group.succession.evidence.v1"
# Careful automatic, two voters: the silence a successor must have seen in both directions.
CAREFUL_SILENCE_SECONDS = 180.0
# Proofs of a move nobody chose: a returning host that kept writing asks the owner which history stays.
AUTOMATIC_PROOFS = frozenset({"certified", "evidence"})
REQUEST_FRESHNESS_SECONDS = 600
MAX_RUN_EVIDENCE = 256
MAX_LABEL_CHARS = 200
RECORDS = "hosted_room_succession"
LINEAGE = "hosted_room_succession_lineage"
CONSENT = "hosted_room_succession_consent"


class SuccessionError(rooms.HostedRoomError):
    """A succession step that cannot proceed; ``reason`` is the stable code, ``detail`` its parameters."""

    reason = "succession_unavailable"

    def __init__(self, message: str, *, reason: str | None = None, detail: Mapping[str, Any] | None = None):
        super().__init__(message)
        if reason is not None:
            self.reason = reason
        self.detail = dict(detail or {})


class ProofInvalid(SuccessionError):
    reason = "succession_proof_invalid"


# --- adapters: #104601 custody and room identity, #99107 verified transitions ---------------------
def _custody():
    from gateway import hosted_room_custody
    return hosted_room_custody


def _identity():
    from gateway import hosted_room_identity
    return hosted_room_identity


def configuration_locked(conn: sqlite3.Connection, room_id: str) -> dict[str, Any]:
    """``{configuration_seq, custodians: [{install_id, public_key, endpoint, role, successor, name,
    operator_name}], owner_name}``."""
    return _custody().configuration_locked(conn, room_id)


def watermark_locked(conn: sqlite3.Connection, room_id: str) -> dict[str, Any] | None:
    """This installation's own durable ``{epoch, seq, event_hash}`` for the room."""
    return _custody().custody_watermark_locked(conn, room_id, store=False)


def reset_chain_locked(conn: sqlite3.Connection, room_id: str, *, after_seq: int) -> None:
    """Forget #104601's derived watermark hashes past ``after_seq`` once history was set aside."""
    _custody().reset_chain_locked(conn, room_id, after_seq=after_seq)


def custody_member_id() -> str:
    """The member id a custodian-only grant carries: it names no Bot."""
    return _custody().CUSTODY_MEMBER_ID


def custody_status(db_path: Path | str, room_id: str) -> dict[str, Any]:
    return _custody().custody_status(db_path, room_id)


def heads_locked(conn: sqlite3.Connection, room_id: str) -> list[dict[str, Any]]:
    """The host-signed heads that vouch for this installation's history, one per epoch (#104601)."""
    return _custody().heads_locked(conn, room_id)


def keep_own_head_locked(conn: sqlite3.Connection, room_id: str) -> dict[str, Any]:
    """On a host about to continue its own group at a fresh epoch: keep its head for the epoch it leaves."""
    return _custody().keep_own_head_locked(conn, room_id)


def automatic_requested_locked(conn: sqlite3.Connection, room_id: str) -> bool:
    """On the host, the owner's latest choice for automatic moves (#104601); the configuration carries
    it once the voters settle."""
    return bool(_custody().automatic_locked(conn, room_id))


def automatic_pending(db_path: Path | str, room_id: str, *, enabled: bool) -> bool:
    """Whether the owner's switch is not yet in force with the voters (#104601)."""
    return bool(_custody().automatic_pending(db_path, room_id, enabled=enabled))


def sign(domain: bytes, payload: Mapping[str, Any]) -> str:
    return _identity().sign(domain, dict(payload))


def verify_locked(conn: sqlite3.Connection, room_id: str, install_id: str, domain: bytes,
                  payload: Mapping[str, Any], signature: Any) -> bool:
    return _identity().verify_locked(conn, room_id, install_id, domain, dict(payload), signature)


def local_consent(db_path: Path | str, room_id: str) -> bool:
    """Whether this computer's own operator still consents to continue this group here."""
    return _custody().local_consent(db_path, room_id) is True


def mark_moved_to_branch(conn: sqlite3.Connection, *, room_id: str, to_epoch: int, branch_id: str) -> dict[str, Any]:
    """Archive a stepped-down successor's own transition mark with its quarantined branch (#99107)."""
    from gateway.hosted_room_safety import move_transition_mark_to_branch
    return move_transition_mark_to_branch(conn, room_id=room_id, to_epoch=to_epoch, branch_id=branch_id)


def local_install_id() -> str:
    return rooms.local_authority_gateway_id()


def proof_digest(proof: Mapping[str, Any]) -> str:
    from gateway.hosted_room_safety import transition_proof_digest
    return transition_proof_digest(dict(proof))


def digest(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()


def nonce() -> str:
    return os.urandom(8).hex()


def transition_text(successor_name: str | None) -> str:
    """The English fallback clients show for the notice; they localize from the event kind."""
    return f"This group now continues on {successor_name or 'another computer'}."


# --- reply sealing: grants and run evidence reach only the requesting successor -------------------
def reply_keypair() -> tuple[Any, str]:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    private = X25519PrivateKey.generate()
    return private, private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def _seal_key(shared: bytes, aad: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_SEAL_INFO + aad).derive(shared)


def seal(public_hex: str, value: Any, *, aad: bytes) -> dict[str, str]:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    ephemeral = X25519PrivateKey.generate()
    shared = ephemeral.exchange(X25519PublicKey.from_public_bytes(bytes.fromhex(public_hex)))
    sealed_nonce = secrets.token_bytes(12)
    data = AESGCM(_seal_key(shared, aad)).encrypt(sealed_nonce, compact_json(value).encode("utf-8"), aad)
    return {"key": ephemeral.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),
            "nonce": base64.b64encode(sealed_nonce).decode("ascii"), "data": base64.b64encode(data).decode("ascii")}


def open_sealed(private: Any, sealed: Any, *, aad: bytes) -> Any:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    try:
        if not isinstance(sealed, dict) or set(sealed) != {"key", "nonce", "data"}:
            raise ValueError("sealed reply has the wrong shape")
        shared = private.exchange(X25519PublicKey.from_public_bytes(bytes.fromhex(sealed["key"])))
        plain = AESGCM(_seal_key(shared, aad)).decrypt(
            base64.b64decode(sealed["nonce"]), base64.b64decode(sealed["data"]), aad)
        return json.loads(plain)
    except (InvalidTag, ValueError, TypeError) as exc:
        raise SuccessionError("a sealed succession reply could not be opened") from exc


# --- configuration ---------------------------------------------------------------------------------
def custodians_by_id(configuration: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {str(c["install_id"]): dict(c) for c in (configuration or {}).get("custodians", ())
            if isinstance(c, Mapping) and isinstance(c.get("install_id"), str)}


def host_entry(configuration: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return next((c for c in custodians_by_id(configuration).values() if c.get("role") == "authority"), None)


def is_eligible(configuration: Mapping[str, Any] | None, install_id: str) -> bool:
    entry = custodians_by_id(configuration).get(install_id)
    return entry is not None and entry.get("successor") is True and entry.get("role") != "authority"


def label(configuration: Mapping[str, Any] | None, install_id: str | None, field: str = "name") -> str | None:
    """A computer's display label (or ``operator_name``) as the room's configuration names it."""
    entry = custodians_by_id(configuration).get(install_id or "")
    return display_label(entry.get(field) if entry else None, max_chars=MAX_LABEL_CHARS)


def owner_label(configuration: Mapping[str, Any] | None) -> str | None:
    return display_label((configuration or {}).get("owner_name"), max_chars=MAX_LABEL_CHARS)


def configuration_through_locked(conn: sqlite3.Connection, room_id: str, seq: int) -> dict[str, Any]:
    """The configuration in force at ``seq``: a host stepping down may hold later, divergent events."""
    for configuration in reversed(_custody().configurations_locked(conn, room_id)):
        if configuration["seq"] <= seq:
            return {"configuration_seq": configuration["seq"],
                    **{key: value for key, value in configuration.items() if key != "seq"}}
    return {"configuration_seq": 0, "custodians": [], "owner_name": None, "automatic": True, "voters": []}


def _unsigned(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "signature"}


# --- fence receipts and the attestation --------------------------------------------------------------
# A custodian keeps one head per epoch its copy went through (#104601): a receipt carries at most these.
MAX_RECEIPT_HEADS = 64


def fence_receipt(*, room_id: str, custodian_install_id: str, fence_state: Mapping[str, Any],
                  watermark: Mapping[str, Any], configuration_seq: int, request_digest: str,
                  evidence_digest: str, heads: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """What a backup signs when it fences: never grants or evidence bodies, only their digests. ``heads``
    are the host-signed heads that vouch for its copy, one per epoch, oldest first: only what they
    vouch for can be adopted from it."""
    return {"room_id": room_id, "custodian_install_id": custodian_install_id,
            "fenced_epoch": int(fence_state["fenced_epoch"]), "promise": dict(fence_state["promise"]),
            "watermark": {key: watermark[key] for key in ("epoch", "seq", "event_hash")},
            "heads": [dict(head) for head in (heads or ())][-MAX_RECEIPT_HEADS:],
            "configuration_seq": int(configuration_seq), "request_digest": request_digest,
            "evidence_digest": evidence_digest}


def vouched_receipts_locked(conn: sqlite3.Connection, room_id: str, receipts: list[Mapping[str, Any]], *,
                            host: str, epoch: int) -> list[dict[str, Any]]:
    """For each receipt, the highest head in it that ``host`` signed for ``epoch``, checked against the
    key pinned here: ``{install_id, seq, head}``. A receipt's own watermark is only its custodian's word."""
    custody = _custody()
    found = []
    for receipt in receipts:
        best = None
        for head in (receipt.get("heads") if isinstance(receipt.get("heads"), list) else ())[-MAX_RECEIPT_HEADS:]:
            try:
                statement = custody.verify_head_locked(conn, room_id, head)
            except Exception:
                continue
            if (statement["room_id"], statement["host"], statement["epoch"]) != (room_id, host, epoch):
                continue
            if best is None or statement["seq"] > best["seq"]:
                best = {"install_id": receipt.get("custodian_install_id"), "seq": int(statement["seq"]),
                        "head": dict(head)}
        if best is not None:
            found.append(best)
    return found


def check_receipt_locked(conn: sqlite3.Connection, room_id: str, signed: Any, *, to_epoch: int,
                         successor: str, custodians: Mapping[str, Any]) -> dict[str, Any]:
    """One genuine fence receipt of ``to_epoch`` for ``successor`` by a configured backup, else ProofInvalid."""
    if not isinstance(signed, dict) or not isinstance(signed.get("promise"), dict):
        raise ProofInvalid("a fence receipt has the wrong shape")
    receipt = _unsigned(signed)
    custodian, promise = receipt.get("custodian_install_id"), receipt["promise"]
    if (receipt.get("room_id") != room_id or custodian not in custodians
            or promise.get("epoch") != to_epoch or promise.get("candidate_install_id") != successor
            or receipt.get("fenced_epoch") != to_epoch - 1 or not isinstance(receipt.get("watermark"), dict)
            or not isinstance(receipt.get("heads", []), list) or len(receipt.get("heads", [])) > MAX_RECEIPT_HEADS):
        raise ProofInvalid("a fence receipt names another room, computer, epoch or successor")
    if not verify_locked(conn, room_id, custodian, FENCE_RECEIPT, receipt, signed.get("signature")):
        raise ProofInvalid("a fence receipt is not signed by its computer's pinned key")
    return receipt


def build_attestation(*, room_id: str, from_epoch: int, to_epoch: int, successor: str, previous_authority: str,
                      origin_install_id: str, configuration_seq: int, receipts: list[dict[str, Any]],
                      unreachable: list[str], confirmed_by: str, preview_id: str | None,
                      decision: Mapping[str, Any] | None = None, statement: str | None = None,
                      stalled: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The successor's signed statement of the owner's decision, with every fence receipt it got.

    With ``decision`` it is the kept host's own continuation at a fresh epoch after the owner chose
    it over a rival successor: everyone can follow that epoch, whichever computer they followed.
    ``statement`` ``ANYWAY_TEXT`` is a paused host the owner continued anyway; ``RECOVER_TEXT`` a host that
    continues past a step it saw promised to ``stalled`` ``{install_id, epoch}``, which never took it.
    """
    unsigned = {"kind": "attestation", "room_id": room_id, "from_epoch": from_epoch, "to_epoch": to_epoch,
                "successor_gateway_id": successor, "previous_authority": previous_authority,
                "origin_install_id": origin_install_id, "configuration_seq": configuration_seq,
                "statement": statement or (KEEP_TEXT if decision is not None else ATTESTATION_TEXT),
                "confirmed_by": confirmed_by, "confirmed_at": time.time(), "preview_id": preview_id,
                "unreachable": sorted(unreachable),
                "receipts": sorted(receipts, key=lambda item: item["custodian_install_id"]),
                **({"decision": dict(decision)} if decision is not None else {}),
                **({"stalled": dict(stalled)} if stalled is not None else {})}
    return {**unsigned, "signature": sign(ATTESTATION, unsigned)}


def verify_decision_locked(conn: sqlite3.Connection, room_id: str, decision: Any) -> dict[str, Any]:
    """The owner's choice after ``continued_on_two``, signed by one of the two computers it names."""
    if not isinstance(decision, dict):
        raise ProofInvalid("the owner's choice has the wrong shape")
    unsigned = _unsigned(decision)
    signer, kept, discarded = (unsigned.get(key) for key in ("signer_install_id", "keep_install_id",
                                                              "discard_install_id"))
    side = unsigned.get("kept") if isinstance(unsigned.get("kept"), dict) else {}
    kept_event, kept_head = side.get("event"), side.get("head")
    if isinstance(kept_event, dict):
        names_kept = ((kept_event.get("payload") or {}).get("successor_gateway_id") == kept
                      and type(unsigned.get("epoch")) is int
                      and (kept_event.get("payload") or {}).get("to_epoch") == unsigned["epoch"])
    else:  # the kept side never moved: the group's earlier host, still at its own epoch
        names_kept = isinstance(kept_head, dict) and kept_head.get("install_id") == kept
    if (unsigned.get("room_id") != room_id or not isinstance(kept, str) or not isinstance(discarded, str)
            or kept == discarded or signer not in {kept, discarded} or type(unsigned.get("epoch")) is not int
            or not names_kept):
        raise ProofInvalid("the owner's choice does not name this group's two computers")
    if not verify_locked(conn, room_id, signer, DECISION, unsigned, decision.get("signature")):
        raise ProofInvalid("the owner's choice is not signed by one of the two computers")
    if unsigned.get("decided_by") == "rule":
        # Without the owner, only the host the rule keeps may sign, and anyone can check the rule; the
        # side passed over checks it against its own head, not the one the decision claims for it.
        from gateway.hosted_room_succession_move import side_of, winner_of
        discarded_side = unsigned.get("discarded") if isinstance(unsigned.get("discarded"), dict) else {}
        own = _own_side_locked(conn, room_id) if discarded == local_install_id() else None
        sides = [side_of(side), own if own is not None else side_of(discarded_side)]
        if signer != kept or None in sides or {item["install_id"] for item in sides} != {kept, discarded} or (
                winner_of(sides) != kept):
            raise ProofInvalid("the rule does not keep that computer")
    return unsigned


def _own_side_locked(conn: sqlite3.Connection, room_id: str) -> dict[str, Any] | None:
    """This computer's own side of a split, from its own store: the epoch it hosts and how it got it."""
    me = local_install_id()
    row = conn.execute("SELECT authority_gateway_id, authority_epoch FROM hosted_rooms WHERE room_id=? "
                       "AND disbanded_at IS NULL", (room_id,)).fetchone()
    if row is None or row[0] != me:
        return None
    mine = latest_transition_locked(conn, room_id)
    payload = mine["event"]["payload"] if mine is not None else {}
    kind = payload.get("proof_kind") if (payload.get("successor_gateway_id") == me
                                         and int(payload.get("to_epoch") or 0) == int(row[1])) else None
    return {"install_id": me, "since": None, "epoch": int(row[1]), "proof_kind": kind}


def verify_claim_locked(conn: sqlite3.Connection, room_id: str, payload: Mapping[str, Any], *,
                        fork_seq: int) -> None:
    """A rival host's signed claim, checked against the history both share.

    It must name an eligible computer of the configuration in force there (or that configuration's
    host continuing its own group), for this room and step, and carry the signature its kind
    requires: the successor's for ``attested``, ``certified`` and ``evidence``, the host's for
    ``handover``. Its receipts and the rest of its proof are verified where its history is ingested.
    """
    proof, kind = payload.get("proof"), payload.get("proof_kind")
    successor = payload.get("successor_gateway_id")
    if not isinstance(proof, dict) or payload.get("proof_digest") != proof_digest(proof) or not isinstance(
            successor, str):
        raise ProofInvalid("the other continuation has the wrong shape")
    configuration = configuration_through_locked(conn, room_id, fork_seq)
    host = (host_entry(configuration) or {}).get("install_id")
    step = {"room_id": room_id, "from_epoch": payload.get("from_epoch"), "to_epoch": payload.get("to_epoch")}
    if kind in {"attested", "certified"}:
        signed, domain, signer = _unsigned(proof), ATTESTATION if kind == "attested" else CERTIFICATE, successor
        named = {**step, "successor_gateway_id": successor}
    elif kind in {"evidence", "handover"}:
        signed = proof.get("statement") if isinstance(proof.get("statement"), dict) else {}
        domain = EVIDENCE if kind == "evidence" else _handover_domain()
        signer, named = (successor if kind == "evidence" else host), {**step, "successor": successor}
    else:
        raise ProofInvalid("the other continuation carries an unknown kind of proof")
    if any(signed.get(key) != value for key, value in named.items()):
        raise ProofInvalid("the other continuation has the wrong shape")
    if not is_eligible(configuration, successor) and successor != host:
        raise ProofInvalid("the other continuation names a computer the owner did not allow")
    if not isinstance(signer, str) or not verify_locked(conn, room_id, signer, domain, signed,
                                                        proof.get("signature")):
        raise ProofInvalid("the other continuation is not signed by its computer's pinned key")


def _handover_domain() -> bytes:
    from gateway.hosted_room_succession_handover import HANDOVER
    return HANDOVER


def voters_of(configuration: Mapping[str, Any] | None) -> list[str]:
    """The group's voting computers, in the owner's order: the host, then its always-on successors."""
    return [item for item in (configuration or {}).get("voters") or () if isinstance(item, str)]


def majority(count: int) -> int:
    return count // 2 + 1


def verify_proof_locked(conn: sqlite3.Connection, room_id: str, *, proof_kind: str, proof: Any,
                        from_epoch: int, to_epoch: int, successor: str, fork_seq: int | None = None,
                        configuration: Mapping[str, Any] | None = None) -> None:
    """Verify a change of host against this installation's own copy of the configuration.

    ``attested``: the owner continued the group (or kept, or continued its paused host anyway).
    ``certified``: a majority of the voters promised, their leases to the host run out.
    ``handover``: the host signed its exact history over. ``evidence``: in a two-voter group, the
    standby's signed statement of the careful silence.
    """
    if configuration is None:
        configuration = configuration_locked(conn, room_id)
    if proof_kind == "handover":
        from gateway.hosted_room_succession_handover import verify_locked as verify_handover
        verify_handover(conn, room_id, proof, from_epoch=from_epoch, to_epoch=to_epoch, successor=successor,
                        fork_seq=fork_seq, configuration=configuration)
        return
    if proof_kind == "evidence":
        _verify_evidence_locked(conn, room_id, proof, from_epoch=from_epoch, to_epoch=to_epoch, successor=successor,
                                fork_seq=fork_seq, configuration=configuration)
        return
    if proof_kind == "certified":
        _verify_certificate_locked(conn, room_id, proof, from_epoch=from_epoch, to_epoch=to_epoch,
                                   successor=successor, fork_seq=fork_seq, configuration=configuration)
        return
    if proof_kind != "attested":
        raise ProofInvalid("the change of host carries an unknown kind of proof")
    _verify_attested_locked(conn, room_id, proof, from_epoch=from_epoch, to_epoch=to_epoch, successor=successor,
                            fork_seq=fork_seq, configuration=configuration)


def _verify_attested_locked(conn, room_id, proof, *, from_epoch, to_epoch, successor, fork_seq, configuration):
    """The successor must be an eligible computer of the configuration and must have signed; its own
    fence receipt and every other one must be genuine and distinct; the host it replaced must be the
    configured host; and it must have adopted the most complete copy among its receipts. A kept host's
    continuation after ``continued_on_two``, or a paused host continued anyway, names itself as both."""
    custodians = custodians_by_id(configuration)
    host = host_entry(configuration)
    expected = {"kind": "attestation", "room_id": room_id, "from_epoch": from_epoch, "to_epoch": to_epoch,
                "successor_gateway_id": successor, "configuration_seq": configuration.get("configuration_seq")}
    if (not isinstance(proof, dict) or any(proof.get(key) != value for key, value in expected.items())
            or to_epoch <= from_epoch or host is None or proof.get("previous_authority") != host["install_id"]
            or not isinstance(proof.get("receipts"), list)):
        raise ProofInvalid("the continuation does not match this room's configuration")
    keeping = proof.get("statement") in {KEEP_TEXT, ANYWAY_TEXT, RECOVER_TEXT}
    stalled = proof.get("stalled") if isinstance(proof.get("stalled"), dict) else {}
    if proof.get("statement") == KEEP_TEXT:
        decision = verify_decision_locked(conn, room_id, proof.get("decision"))
        if successor != host["install_id"] or decision["keep_install_id"] != successor:
            raise ProofInvalid("only the kept host continues its own group after the owner's choice")
    elif proof.get("statement") == ANYWAY_TEXT:
        if successor != host["install_id"]:
            raise ProofInvalid("only the paused host itself is continued anyway")
    elif proof.get("statement") == RECOVER_TEXT:
        if (successor != host["install_id"] or not isinstance(stalled.get("install_id"), str)
                or stalled["install_id"] == successor or stalled["install_id"] not in custodians):
            raise ProofInvalid("only the host continues past a step another computer of the group never took")
    elif proof.get("statement") != ATTESTATION_TEXT or not is_eligible(configuration, successor):
        raise ProofInvalid("the successor is not a computer the owner allowed to continue this group")
    if not verify_locked(conn, room_id, successor, ATTESTATION, _unsigned(proof), proof.get("signature")):
        raise ProofInvalid("the continuation is not signed by its successor's pinned key")
    receipts = [check_receipt_locked(conn, room_id, item, to_epoch=to_epoch, successor=successor,
                                     custodians=custodians) for item in proof["receipts"]]
    fenced = [item["custodian_install_id"] for item in receipts]
    if len(set(fenced)) != len(fenced) or successor not in fenced:
        raise ProofInvalid("the continuation's fence receipts are duplicated or miss the successor's own")
    if proof.get("statement") == RECOVER_TEXT:
        # Every computer that could have continued the group (each eligible successor, and the one that
        # never took the step) promised this fresh step instead, so no step taken or still pending runs
        # beside it; in majority mode a majority of the voters did too.
        could = {install_id for install_id in custodians if is_eligible(configuration, install_id)}
        voters = voters_of(configuration)
        if ((could | {stalled["install_id"]}) - {successor} - set(fenced)) or (
                _custody().mode_of(configuration) == "majority"
                and len(set(fenced) & set(voters)) < majority(len(voters))):
            raise ProofInvalid("every computer that could have continued the group must have promised this step")
    if not keeping and fork_seq is not None and any(item["seq"] > fork_seq for item in vouched_receipts_locked(
            conn, room_id, receipts, host=host["install_id"], epoch=from_epoch)):
        raise ProofInvalid("the successor did not adopt the most complete reachable copy")


def build_certificate(*, room_id: str, from_epoch: int, to_epoch: int, successor: str, previous_authority: str,
                      origin_install_id: str, configuration: Mapping[str, Any],
                      receipts: list[dict[str, Any]]) -> dict[str, Any]:
    """The successor's certificate: signed promises from a majority of the voters, leases run out."""
    unsigned = {"kind": "certificate", "room_id": room_id, "from_epoch": from_epoch, "to_epoch": to_epoch,
                "successor_gateway_id": successor, "previous_authority": previous_authority,
                "origin_install_id": origin_install_id,
                "configuration_seq": configuration.get("configuration_seq"), "voters": voters_of(configuration),
                "issued_at": time.time(),
                "receipts": sorted(receipts, key=lambda item: item["custodian_install_id"])}
    return {**unsigned, "signature": sign(CERTIFICATE, unsigned)}


def _verify_certificate_locked(conn, room_id, proof, *, from_epoch, to_epoch, successor, fork_seq, configuration):
    host, voters = host_entry(configuration), voters_of(configuration)
    expected = {"kind": "certificate", "room_id": room_id, "from_epoch": from_epoch, "to_epoch": to_epoch,
                "successor_gateway_id": successor, "configuration_seq": configuration.get("configuration_seq"),
                "voters": voters}
    if (not isinstance(proof, dict) or any(proof.get(key) != value for key, value in expected.items())
            or host is None or proof.get("previous_authority") != host["install_id"] or len(voters) < 3
            or successor not in voters or not is_eligible(configuration, successor)
            or not isinstance(proof.get("receipts"), list)):
        raise ProofInvalid("the certificate does not match this group's voters")
    if not verify_locked(conn, room_id, successor, CERTIFICATE, _unsigned(proof), proof.get("signature")):
        raise ProofInvalid("the certificate is not signed by its successor's pinned key")
    custodians = custodians_by_id(configuration)
    receipts = [check_receipt_locked(conn, room_id, item, to_epoch=to_epoch, successor=successor,
                                     custodians=custodians) for item in proof["receipts"]]
    promised = {item["custodian_install_id"] for item in receipts} & set(voters)
    if len(promised) < majority(len(voters)) or successor not in promised or len(receipts) != len(
            {item["custodian_install_id"] for item in receipts}):
        raise ProofInvalid("the certificate lacks promises from a majority of the voters")
    if fork_seq is not None and any(item["seq"] > fork_seq for item in vouched_receipts_locked(
            conn, room_id, receipts, host=host["install_id"], epoch=from_epoch)):
        raise ProofInvalid("the successor did not adopt the most complete promised copy")


def evidence_statement(conn: sqlite3.Connection, room_id: str, *, from_epoch: int, to_epoch: int, successor: str,
                       silent_since: float, silent_for_s: float) -> dict[str, Any]:
    mark = watermark_locked(conn, room_id)
    return {"room_id": room_id, "from_epoch": from_epoch, "to_epoch": to_epoch, "successor": successor,
            "last_seq": int(mark["seq"]), "last_hash": mark["event_hash"], "silent_since": float(silent_since),
            "silent_for_s": float(silent_for_s)}


def _verify_evidence_locked(conn, room_id, proof, *, from_epoch, to_epoch, successor, fork_seq, configuration):
    host, voters = host_entry(configuration), voters_of(configuration)
    statement = proof.get("statement") if isinstance(proof, Mapping) else None
    expected = {"room_id": room_id, "from_epoch": from_epoch, "to_epoch": to_epoch, "successor": successor}
    if (not isinstance(statement, Mapping) or any(statement.get(key) != value for key, value in expected.items())
            or host is None or voters != [host["install_id"], successor] or not is_eligible(configuration, successor)
            or (configuration or {}).get("automatic") is False
            or not isinstance(statement.get("silent_for_s"), (int, float))
            or statement["silent_for_s"] < CAREFUL_SILENCE_SECONDS
            or (fork_seq is not None and statement.get("last_seq") != fork_seq)):
        raise ProofInvalid("the evidence does not match this two-voter group")
    if not verify_locked(conn, room_id, successor, EVIDENCE, dict(statement), proof.get("signature")):
        raise ProofInvalid("the evidence is not signed by its successor's pinned key")
    if fork_seq is not None and chain_hash_or_none(conn, room_id, int(fork_seq)) != statement.get("last_hash"):
        raise ProofInvalid("the evidence names a history this computer does not hold")


def chain_hash_or_none(conn: sqlite3.Connection, room_id: str, seq: int) -> str | None:
    """The custody chain hash of this computer's history through ``seq``, or ``None`` short of it."""
    try:
        return _custody().chain_hash_locked(conn, room_id, seq, store=False)
    except rooms.HostedRoomError:
        return None


def transition_for(proof: Mapping[str, Any], kind: str = "attested") -> dict[str, Any]:
    """The ``transition`` argument #99107's promote_replica/demote_room take."""
    return {"proof_kind": kind, "proof_digest": proof_digest(proof), "proof": dict(proof)}


def verify_transition_locked(conn: sqlite3.Connection, event: Mapping[str, Any]) -> None:
    """``_verify_transition`` for custody ingest: verify the event's proof, mark it, learn its lineage.

    Runs inside the copy's writer for each ``authority.transition`` before its insert. A refusal
    leaves the copy where it was; it never quarantines a half-written copy.
    """
    from gateway.hosted_room_safety import mark_verified_transition
    try:
        payload = json.loads(event["payload_json"])
        room_id, proof, kind = str(event["room_id"]), payload["proof"], payload["proof_kind"]
        from_epoch, to_epoch = int(payload["from_epoch"]), int(payload["to_epoch"])
        successor = str(payload["successor_gateway_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProofInvalid("the authority transition has the wrong shape") from exc
    if payload.get("proof_digest") != proof_digest(proof) or int(event["authority_epoch"]) != to_epoch:
        raise ProofInvalid("the authority transition does not bind its proof")
    from gateway.hosted_room_succession_automatic import db_file, may_follow
    store = db_file(conn)
    if store is not None and not may_follow(store, room_id, to_epoch, successor):
        raise ProofInvalid("this computer already moved past that epoch with another computer")
    verify_proof_locked(conn, room_id, proof_kind=kind, proof=proof, from_epoch=from_epoch, to_epoch=to_epoch,
                        successor=successor, fork_seq=int(event["seq"]) - 1)
    mark_verified_transition(conn, room_id=room_id, from_epoch=from_epoch, to_epoch=to_epoch,
                             successor_gateway_id=successor, proof_kind=kind, proof_digest=payload["proof_digest"])
    record_lineage_locked(conn, room_id, origin_install_id=str(proof.get("origin_install_id") or ""),
                          gateway_id=successor, epoch=to_epoch, role=kind, proof_digest=payload["proof_digest"])


def latest_transition_locked(conn: sqlite3.Connection, room_id: str,
                             from_epoch: int | None = None) -> dict[str, Any] | None:
    """The verified ``authority.transition`` held here for the room, with the event just before it: the
    one leaving ``from_epoch`` when given (a returning host's own epoch), else the latest."""
    for table in ("hosted_room_events", "hosted_room_replica_events"):
        row = conn.execute(f"""SELECT event.seq, event.event_id, event.kind, event.actor_json, event.authority_epoch,
                event.payload_json, event.created_at FROM {table} AS event
              JOIN hosted_room_verified_transition_uses AS used ON used.room_id=event.room_id
                   AND used.seq=event.seq AND used.event_id=event.event_id
             WHERE event.room_id=? AND event.kind='authority.transition'
               AND (? IS NULL OR json_extract(event.payload_json, '$.from_epoch')=?)
             ORDER BY event.seq DESC LIMIT 1""", (room_id, from_epoch, from_epoch)).fetchone()
        if row is not None:
            fork = conn.execute(f"""SELECT seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at
                FROM {table} WHERE room_id=? AND seq=?""", (room_id, int(row[0]) - 1)).fetchone()
            return {"event": event_dict(room_id, row), "fork_event": event_dict(room_id, fork) if fork else None}
    return None


def event_dict(room_id: str, row) -> dict[str, Any]:
    return {"room_id": room_id, "seq": int(row[0]), "event_id": row[1], "kind": row[2], "actor": json.loads(row[3]),
            "authority_epoch": int(row[4]), "payload": json.loads(row[5]), "created_at": float(row[6])}


# --- records: per-room state, learned lineage, continuation consent -------------------------------
def owner_subject_locked(conn: sqlite3.Connection, room_id: str) -> str | None:
    """The subject this computer recorded as the room's owner: the owner key where it hosts the room,
    else the principal that consented here to continue it."""
    if table_exists(conn, "state_meta"):
        row = conn.execute("SELECT value FROM state_meta WHERE key=?",
                           ("gateway.hosted.owner.v1:" + room_id,)).fetchone()
        if row is not None:
            return str(row[0])
    record = load_record_locked(conn, room_id, "owner")
    return str(record["subject"]) if record and isinstance(record.get("subject"), str) else None


def record_owner_subject(db_path: Path | str, room_id: str, subject: Any) -> None:
    """The principal that consented here to continue the room, or to keep a copy of it, becomes its
    owner on this computer, unless the room already has one here (it hosts the room, or someone
    consented first). The owner key is the one ``groups.log``, the copy readers and messaging read; it
    stays through moves and stepping down. Creating a hosted room never reuses it for a held copy."""
    if not isinstance(subject, str) or not subject:
        return
    with rooms._transaction(Path(db_path), immediate=True) as conn:
        if table_exists(conn, "state_meta") and conn.execute(
                "SELECT 1 FROM state_meta WHERE key=?", ("gateway.hosted.owner.v1:" + room_id,)).fetchone():
            return
        save_record_locked(conn, room_id, "owner", {"subject": subject, "recorded_at": time.time()})
        if table_exists(conn, "state_meta"):
            conn.execute("INSERT OR IGNORE INTO state_meta(key, value) VALUES (?, ?)",
                         ("gateway.hosted.owner.v1:" + room_id, subject))


def is_owner_locked(conn: sqlite3.Connection, room_id: str, *, subject: str | None, operator: bool) -> bool:
    """The caller may act for the room's owner here: this computer's operator, or the recorded owner."""
    return operator is True or (subject is not None and subject == owner_subject_locked(conn, room_id))


def initialize_succession_schema(conn: sqlite3.Connection) -> None:
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {RECORDS} (
        room_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('move', 'conflict', 'heartbeat', 'owner', 'return', 'notice', 'automatic')),
        record_json TEXT NOT NULL, updated_at REAL NOT NULL, PRIMARY KEY (room_id, kind))""")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {LINEAGE} (
        room_id TEXT NOT NULL, epoch INTEGER NOT NULL, gateway_id TEXT NOT NULL, role TEXT NOT NULL,
        origin_install_id TEXT NOT NULL, proof_digest TEXT, recorded_at REAL NOT NULL,
        PRIMARY KEY (room_id, epoch, gateway_id, role))""")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {CONSENT} (
        room_id TEXT NOT NULL, member_id TEXT NOT NULL, target_profile TEXT NOT NULL,
        options_json TEXT NOT NULL, consented_at REAL NOT NULL,
        PRIMARY KEY (room_id, member_id, target_profile))""")
    # A learned or promised host is evidence: it never changes or disappears.
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_succession_lineage_kept
        BEFORE DELETE ON {LINEAGE} BEGIN SELECT RAISE(ABORT, 'succession lineage is kept'); END""")
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_succession_lineage_immutable
        BEFORE UPDATE ON {LINEAGE} BEGIN SELECT RAISE(ABORT, 'succession lineage is kept'); END""")


def _ready(conn: sqlite3.Connection) -> None:
    if not all(table_exists(conn, table) for table in (RECORDS, LINEAGE, CONSENT)):
        initialize_succession_schema(conn)


def load_record_locked(conn: sqlite3.Connection, room_id: str, kind: str) -> dict[str, Any] | None:
    if not table_exists(conn, RECORDS):
        return None
    row = conn.execute(f"SELECT record_json FROM {RECORDS} WHERE room_id=? AND kind=?", (room_id, kind)).fetchone()
    return json.loads(row[0]) if row is not None else None


def save_record_locked(conn: sqlite3.Connection, room_id: str, kind: str, record: Mapping[str, Any]) -> None:
    _ready(conn)
    conn.execute(f"""INSERT INTO {RECORDS}(room_id, kind, record_json, updated_at) VALUES (?,?,?,?)
        ON CONFLICT(room_id, kind) DO UPDATE SET record_json=excluded.record_json, updated_at=excluded.updated_at""",
                 (room_id, kind, compact_json(dict(record)), time.time()))


def load_record(db_path: Path | str, room_id: str, kind: str) -> dict[str, Any] | None:
    with closing(rooms._read_connection(Path(db_path))) as conn:
        return load_record_locked(conn, room_id, kind)


def save_record(db_path: Path | str, room_id: str, kind: str, record: Mapping[str, Any]) -> None:
    with rooms._transaction(Path(db_path), immediate=True) as conn:
        save_record_locked(conn, room_id, kind, record)


def record_lineage_locked(conn: sqlite3.Connection, room_id: str, *, origin_install_id: str, gateway_id: str,
                          epoch: int, role: str, proof_digest: str | None = None) -> None:
    _ready(conn)
    if not origin_install_id:
        origin_install_id = origin_locked(conn, room_id) or gateway_id
    conn.execute(f"""INSERT OR IGNORE INTO {LINEAGE}
        (room_id, epoch, gateway_id, role, origin_install_id, proof_digest, recorded_at) VALUES (?,?,?,?,?,?,?)""",
        (room_id, epoch, gateway_id, role, origin_install_id, proof_digest, time.time()))


def log_origin_locked(conn: sqlite3.Connection, room_id: str, current_authority: str) -> str:
    """The original home, from the latest transition this installation holds, else its current host."""
    for table in ("hosted_room_events", "hosted_room_replica_events"):
        row = conn.execute(f"""SELECT payload_json FROM {table} WHERE room_id=? AND kind='authority.transition'
            ORDER BY seq DESC LIMIT 1""", (room_id,)).fetchone()
        if row is not None:
            origin = json.loads(row[0]).get("proof", {}).get("origin_install_id")
            if isinstance(origin, str) and origin:
                return origin
    return current_authority


def origin_locked(conn: sqlite3.Connection, room_id: str) -> str | None:
    """The room's original home, once this installation saw its host change or be promised away."""
    if not table_exists(conn, LINEAGE):
        return None
    row = conn.execute(f"SELECT origin_install_id FROM {LINEAGE} WHERE room_id=? ORDER BY epoch LIMIT 1",
                       (room_id,)).fetchone()
    return str(row[0]) if row is not None else None


def lineage_locked(conn: sqlite3.Connection, room_id: str) -> list[dict[str, Any]]:
    if not table_exists(conn, LINEAGE):
        return []
    return [{"epoch": int(r[0]), "gateway_id": r[1], "role": r[2], "origin_install_id": r[3], "proof_digest": r[4],
             "recorded_at": float(r[5])} for r in conn.execute(
        f"""SELECT epoch, gateway_id, role, origin_install_id, proof_digest, recorded_at FROM {LINEAGE}
            WHERE room_id=? ORDER BY epoch, role""", (room_id,))]


def session_home(conn: sqlite3.Connection, room_id: str, home_install_id: str) -> str:
    """The home a participant keys its hidden member session to: the room's origin for any host
    in its learned lineage, so a successor continues the same conversation."""
    if not table_exists(conn, LINEAGE):
        return home_install_id
    known = conn.execute(f"SELECT origin_install_id FROM {LINEAGE} WHERE room_id=? AND gateway_id=? LIMIT 1",
                         (room_id, home_install_id)).fetchone()
    return str(known[0]) if known is not None else home_install_id


def _read_only(db_path: Path | str):
    return closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True, timeout=5))


def member_session_home(db_path: Path | str, *, room_id: str, home_install_id: str) -> str:
    """A participant's member-session home for a dispatch: exactly the dispatching home for a room
    whose host never changed, the room's original home for a verified successor."""
    try:
        with _read_only(db_path) as conn:
            return session_home(conn, room_id, home_install_id)
    except sqlite3.Error:
        return home_install_id


def member_session_for(home_install_id: str, room_id: str, member_id: str, target_profile: str) -> str:
    """The hidden member session a participant keeps for one room member, keyed to a home."""
    seed = f"{home_install_id}\0{room_id}\0{member_id}\0{target_profile}"
    return f"room_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"


def member_session_id(db_path: Path | str, *, home_install_id: str, room_id: str, member_id: str,
                      target_profile: str) -> str:
    home = member_session_home(db_path, room_id=room_id, home_install_id=home_install_id)
    return member_session_for(home, room_id, member_id, target_profile)


def inherited_origin(db_path: Path | str, room_id: str) -> str | None:
    """The original home of a room this installation hosts as a successor, else ``None``."""
    try:
        with _read_only(db_path) as conn:
            origin = origin_locked(conn, room_id)
    except sqlite3.Error:
        return None
    return origin if origin and origin != local_install_id() else None


def record_consent_locked(conn: sqlite3.Connection, *, room_id: str, member_id: str, target_profile: str,
                          options: Mapping[str, Any]) -> None:
    """The participant operator's consent, given with each invitation unless it opts out, to the same
    grant for a verified successor of this room; ``options`` are the invitation's own (copy, evidence,
    lifetimes), never wider."""
    _ready(conn)
    conn.execute(f"""INSERT INTO {CONSENT}(room_id, member_id, target_profile, options_json, consented_at)
        VALUES (?,?,?,?,?) ON CONFLICT(room_id, member_id, target_profile) DO UPDATE SET
        options_json=excluded.options_json, consented_at=excluded.consented_at""",
        (room_id, member_id, target_profile, compact_json(dict(options)), time.time()))


def withdraw_consent_locked(conn: sqlite3.Connection, *, room_id: str, member_id: str, target_profile: str) -> None:
    """``continuation: false`` on a later invitation: this member's grant is not re-issued again."""
    if table_exists(conn, CONSENT):
        conn.execute(f"DELETE FROM {CONSENT} WHERE room_id=? AND member_id=? AND target_profile=?",
                     (room_id, member_id, target_profile))


def consents_locked(conn: sqlite3.Connection, room_id: str) -> list[dict[str, Any]]:
    if not table_exists(conn, CONSENT):
        return []
    return [{"member_id": r[0], "target_profile": r[1], "options": json.loads(r[2])} for r in conn.execute(
        f"SELECT member_id, target_profile, options_json FROM {CONSENT} WHERE room_id=? ORDER BY member_id",
        (room_id,))]


def reconfigure_after_transition_locked(conn: sqlite3.Connection, room_id: str, *, successor: str,
                                        previous_host: str) -> dict[str, Any]:
    """The successor's next ``custody.configured`` (#104601), appended inside the caller's writer."""
    return _custody().reconfigure_after_transition_locked(conn, room_id, successor=successor,
                                                          previous_host=previous_host)


def event_at_locked(conn: sqlite3.Connection, room_id: str, seq: int) -> dict[str, Any] | None:
    """This computer's own event at ``seq``, from the room it hosts or the copy it keeps."""
    for table in ("hosted_room_events", "hosted_room_replica_events"):
        row = conn.execute(f"""SELECT seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at
            FROM {table} WHERE room_id=? AND seq=?""", (room_id, seq)).fetchone()
        if row is not None:
            return event_dict(room_id, row)
    return None


def own_transition_locked(conn: sqlite3.Connection, room_id: str, *, from_epoch: int) -> dict[str, Any] | None:
    """The transition out of ``from_epoch`` this computer's own history holds, if any."""
    for table in ("hosted_room_events", "hosted_room_replica_events"):
        row = conn.execute(f"""SELECT seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at
            FROM {table} WHERE room_id=? AND kind='authority.transition'
             AND json_extract(payload_json, '$.from_epoch')=? ORDER BY seq DESC LIMIT 1""",
                           (room_id, from_epoch)).fetchone()
        if row is not None:
            return event_dict(room_id, row)
    return None


def same_event(mine: Mapping[str, Any] | None, theirs: Mapping[str, Any] | None) -> bool:
    return mine is not None and theirs is not None and all(
        mine.get(key) == theirs.get(key) for key in ("seq", "event_id", "kind", "authority_epoch", "payload"))


# --- the host's own view: lineage for policy, pauses, and waiting work ------------------------------
class WaitingForHostError(RuntimeError):
    """A turn for a Bot that runs only on the group's previous host: deferred with proof it never ran."""

    not_admitted = True
    ambiguous = False
    defer_reason = "waiting_for_host"

    def __init__(self, *, resource: str, host_name: str | None):
        super().__init__("This Bot runs on the group's previous host, which is unavailable.")
        self.defer_detail = {"resource": resource, "host_name": host_name}


def authority_lineage(db_path: Path | str, room_id: str) -> dict[str, str]:
    """``{epoch: host}`` for the epochs before the current one of a room whose host changed; ``{}`` when
    it never did. Each verified transition names the host it replaced and its successor."""
    lineage: dict[str, str] = {}
    try:
        with _read_only(db_path) as conn:
            for (payload_json,) in conn.execute("""SELECT event.payload_json FROM hosted_room_events AS event
                    JOIN hosted_room_verified_transition_uses AS used ON used.room_id=event.room_id
                         AND used.seq=event.seq AND used.event_id=event.event_id
                    WHERE event.room_id=? AND event.kind='authority.transition' ORDER BY event.seq""", (room_id,)):
                payload = json.loads(payload_json)
                previous = (payload.get("proof") or {}).get("previous_authority")
                if isinstance(previous, str) and str(payload["from_epoch"]) not in lineage:
                    lineage[str(payload["from_epoch"])] = previous
                lineage[str(payload["to_epoch"])] = str(payload["successor_gateway_id"])
    except (sqlite3.Error, KeyError, TypeError, ValueError):
        return {}
    return lineage


def paused_reason(db_path: Path | str, room_id: str) -> str | None:
    """Why this host executes and appends nothing for the room right now, else ``None``."""
    try:
        with _read_only(db_path) as conn:
            conflict = load_record_locked(conn, room_id, "conflict") or {}
            returned = load_record_locked(conn, room_id, "return") or {}
            move = load_record_locked(conn, room_id, "move") or {}
    except sqlite3.Error:
        return None
    if conflict.get("state") == "active" and conflict.get("winner") != local_install_id():
        return "room_authority_conflict"  # the other host keeps running until the owner chooses
    if returned.get("state") == "paused" or move.get("state") == "handing_over":
        return "room_authority_promised"
    from gateway.hosted_room_succession_automatic import fenced_here, host_paused_reason
    epoch = _hosted_epoch(db_path, room_id)
    if epoch is not None and fenced_here(db_path, room_id, epoch):
        return "room_authority_promised"
    if epoch is not None and host_paused_reason(db_path, room_id) is not None:
        return "room_host_paused"  # paused to stay safe: no lease from a majority, cut off, or no lease layer
    return None


def _hosted_epoch(db_path: Path | str, room_id: str) -> int | None:
    """The epoch at which this computer hosts the room, if it does."""
    try:
        with _read_only(db_path) as conn:
            row = conn.execute("SELECT authority_gateway_id, authority_epoch FROM hosted_rooms WHERE room_id=? "
                               "AND disbanded_at IS NULL", (room_id,)).fetchone()
    except sqlite3.Error:
        return None
    return int(row[1]) if row is not None and row[0] == local_install_id() else None


def waiting_tasks(db_path: Path | str, room_id: str) -> list[dict[str, Any]]:
    """``driver_status.tasks[]``: the room's turns waiting for a computer that holds what they need."""
    from gateway import hosted_room_driver as driver
    found = []
    try:
        tasks = driver.list_tasks(db_path, room_id=room_id, status="deferred")
    except Exception:
        return []
    for task in tasks:
        result = task.get("result") if isinstance(task.get("result"), dict) else {}
        if result.get("reason") != "waiting_for_host":
            continue
        found.append({"task_id": task["identity"].task_id,
                      "member_id": task["payload"].get("target_member_id") or task["payload"]["target_profile"],
                      "state": "waiting_for_host", "resource": result.get("resource") or "bot",
                      "host_name": result.get("host_name")})
    return found


def chain_after_locked(conn: sqlite3.Connection, room_id: str, epoch: int, *, limit: int = 64) -> list[dict[str, Any]]:
    """The verified ``authority.transition`` and ``custody.configured`` events after ``epoch``, in log
    order: what a computer still at ``epoch`` needs to learn it was superseded (``learn``)."""
    for table in ("hosted_room_events", "hosted_room_replica_events"):
        start = conn.execute(f"""SELECT MIN(event.seq) FROM {table} AS event
            JOIN hosted_room_verified_transition_uses AS used ON used.room_id=event.room_id AND used.seq=event.seq
                 AND used.event_id=event.event_id
            WHERE event.room_id=? AND event.kind='authority.transition'
              AND json_extract(event.payload_json, '$.from_epoch')>=?""", (room_id, epoch)).fetchone()
        if start is not None and start[0] is not None:
            rows = conn.execute(f"""SELECT seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at
                FROM {table} WHERE room_id=? AND (seq>=? OR seq=?-1)
                AND (kind IN ('authority.transition', 'custody.configured') OR seq=?-1)
                ORDER BY seq LIMIT ?""", (room_id, start[0], start[0], start[0], limit)).fetchall()
            return [event_dict(room_id, row) for row in rows]
    return []
