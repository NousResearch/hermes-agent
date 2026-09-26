"""Landing semantics for kanban cards — what makes a card's deliverable *landed*.

A card can read ``done`` on evidence that is not a landing: "designed, implemented
and tested" closes a card whose change never reached the target branch, and a chain
head closes while its own landing leg is still queued behind it. This module is the
single, deterministic answer to two questions the completion path asks:

* **Is this card's deliverable a landing?** — :func:`landing_declaration`. The
  answer never comes from prose a human (or a model) would have to interpret: it
  comes from the board's own structured declarations — the card's
  ``completion_contract`` (the vocabulary ``kanban_pr_acceptance`` already gates),
  the completion's own landing claim, and a non-``done`` child that carries a
  landing contract (the chain-head case). A card that declares none of them is not
  a landing card and its completion is untouched.
* **Does the completion carry the landed evidence?** — :func:`landed_evidence`,
  over the three forms the standing Feature DoD accepts: the merged commit on the
  target branch, the deploy stamp naming it, or a live hash equal to the merged
  blob. A published-but-unmerged PR is not a landing, and neither is a green check
  run. Where a form names a target, the target is READ rather than taken on the
  claim's word: a deploy stamp is opened and the commit it names is what counts
  (a stamp that names no commit is not evidence), and an artifact the live hash
  points at is hashed on disk. A target that contradicts the claim it is offered
  for — a stamp naming a different commit, an artifact hashing to something else —
  refuses the completion outright.

The declaration is the whole mechanism, so it is worth stating plainly: a card's
deliverable is a landing when it declares one of the three structured things above
— a ``completion_contract`` naming a repo or PR (the vocabulary
``kanban_pr_acceptance`` already gates), a landing claim on the completion, or a
child that carries a landing contract and has not landed. A card that declares
none of them is untouched: inferring "this card probably had to land" from its
title or its summary is exactly the prose reading this module exists to avoid, and
it is what makes the ordinary local-only card, the review flow and the chained
landing leg all work without a human in the loop.

Pure functions over a task row / metadata / one bounded child query, so the rules
are testable without a dispatcher.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Optional

#: The one contract value that declares no published deliverable.
LOCAL_ONLY_CONTRACT = "local-only"

#: Metadata keys whose presence is a landing claim on the completion itself.
#: Deliberately tight: a key that unrelated tooling might set as a report field
#: (a timestamp, the ``landing_evidence`` boolean the diagnostics event carries)
#: must not read as a claim, or a non-landing card would be refused.
CLAIM_KEYS = (
    "landed",
    "landed_evidence",
    "merged_commit",
    "merge_commit",
    "deploy_stamp",
    "deploy-stamp",
    "live_sha256",
    "merged_blob_sha256",
)

# The three accepted evidence forms, phrased once so the refusal can name exactly
# what is missing.
EVIDENCE_MERGED_COMMIT = "the merged commit on the target branch"
EVIDENCE_DEPLOY_STAMP = "the deploy stamp naming it"
EVIDENCE_LIVE_HASH = "a live hash equal to the merged blob"
#: Named separately: a live hash that disagrees with its own merged blob.
EVIDENCE_LIVE_HASH_MISMATCH = "a live hash equal to the merged blob (the two hashes disagree)"
#: Named separately: a deploy stamp that names a commit other than the claim's.
EVIDENCE_STAMP_MISMATCH = (
    "a deploy stamp naming the merged commit (the stamp names a different commit)"
)

#: Keys carried into a claim only when another key already makes it one: the
#: target a live hash is read from. Never a claim trigger on their own — a
#: report field that happens to be called ``artifact_path`` must not refuse a
#: completion that never claimed a landing.
CLAIM_CARRIED_KEYS = ("artifact_path", "live_path")

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# A commit named by a stamp's own ``commit:`` line, or a bare token.
_STAMP_COMMIT_LINE_RE = re.compile(r"^[ \t]*commit[ \t]*:[ \t]*([0-9a-fA-F]{7,40})[ \t]*$", re.M)
_COMMIT_TOKEN_RE = re.compile(r"\b[0-9a-fA-F]{7,40}\b")
#: A stamp is read from the target it names — bounded, and never a directory walk.
_STAMP_MAX_BYTES = 64 * 1024
#: The largest artifact whose hash is re-read from disk.
_ARTIFACT_MAX_BYTES = 64 * 1024 * 1024
#: The file a deploy writes its provenance into, under the deployed tree.
_STAMP_BASENAME = ".deployed-from"
# ``owner/repo`` (the same shape kanban_pr_acceptance accepts) or a PR URL.
_CONTRACT_RE = re.compile(r"^[^\s/]+/[^\s/]+$|^https://github\.com/[^\s/]+/[^\s/]+/pull/\d+$")


def _field(task: Any, name: str, default: Any = None) -> Any:
    """Read a field from a sqlite3.Row, a dict, or a task dataclass."""
    if task is None:
        return default
    try:
        if hasattr(task, "keys") and name in task.keys():
            return task[name]
    except Exception:
        pass
    if isinstance(task, dict):
        return task.get(name, default)
    return getattr(task, name, default)


def declares_landing_contract(task: Any) -> bool:
    """True when the card's ``completion_contract`` names a published deliverable.

    ``local-only`` (and NULL) declare nothing; ``OWNER/REPO`` and a PR URL declare
    a deliverable that has to exist outside the card before the card is done.
    """
    contract = _field(task, "completion_contract")
    if not isinstance(contract, str):
        return False
    contract = contract.strip()
    if not contract or contract == LOCAL_ONLY_CONTRACT:
        return False
    return bool(_CONTRACT_RE.match(contract))


def _stamp_text(value: str) -> Optional[str]:
    """The text of the stamp the value names, or ``None`` when it names no file.

    The target is read, never taken on the claim's word: a value pointing at a
    deploy stamp — or at a deployed tree holding one — is opened, and a value
    pointing at nothing is treated as the stamp's own line.
    """
    raw = value.strip()
    if not raw or "\x00" in raw:
        return None
    try:
        path = Path(raw).expanduser()
        if path.is_dir():
            path = path / _STAMP_BASENAME
        if not path.is_file():
            return None
        with path.open("rb") as handle:
            return handle.read(_STAMP_MAX_BYTES).decode("utf-8", "replace")
    except (OSError, ValueError):
        return None


def _is_commit(token: str, *, trust_label: bool = False) -> bool:
    """Is ``token`` a commit sha?

    A ``commit:`` label is trusted to name a commit. A bare token must be a full
    40-char sha or carry a hex letter, so a date or a counter in the stamp text
    (``20260925``) is never read as the commit a stamp names.
    """
    token = token.strip().lower()
    if not _GIT_SHA_RE.match(token):
        return False
    return trust_label or len(token) == 40 or any(ch in "abcdef" for ch in token)


def resolve_deploy_stamp(value: str) -> Optional[str]:
    """The commit the deploy stamp names, read from the target the value points at.

    ``None`` when nothing in the stamp names a commit — a stamp that says only
    *when* something was deployed carries no landing evidence.
    """
    text = _stamp_text(value)
    haystack = text if text is not None else value
    labelled = _STAMP_COMMIT_LINE_RE.search(haystack)
    if labelled and _is_commit(labelled.group(1), trust_label=True):
        return labelled.group(1).lower()
    for match in _COMMIT_TOKEN_RE.finditer(haystack):
        if _is_commit(match.group(0)):
            return match.group(0).lower()
    return None


def _file_sha256(value: str) -> Optional[str]:
    """sha256 of the artifact the value names, or ``None`` when it is unreadable."""
    raw = value.strip()
    if not raw or "\x00" in raw:
        return None
    try:
        path = Path(raw).expanduser()
        if not path.is_file() or path.stat().st_size > _ARTIFACT_MAX_BYTES:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError):
        return None


def landing_claim(metadata: Any) -> Optional[dict]:
    """The completion's own landing claim, or ``None``.

    ``metadata`` may carry the evidence directly (``merged_commit``, …) or under a
    ``landed``/``landed_evidence``/``landing_evidence`` key holding either a
    mapping of evidence or a bare commit sha. A truthy claim that names none of
    the evidence forms is returned as ``{}`` — claimed, unsubstantiated.
    """
    if not isinstance(metadata, dict):
        return None
    found = {key: metadata[key] for key in CLAIM_KEYS if key in metadata}
    if not found:
        return None
    claim: dict = {}
    for key in ("merged_commit", "merge_commit", "deploy_stamp", "deploy-stamp",
                "live_sha256", "merged_blob_sha256"):
        if key in found:
            claim[key] = found[key]
    for key in ("landed", "landed_evidence"):
        if key not in found:
            continue
        value = found[key]
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                claim.setdefault(inner_key, inner_value)
        elif isinstance(value, str) and _GIT_SHA_RE.match(value.strip()):
            claim.setdefault("merged_commit", value.strip())
        elif value:
            # A claim we cannot read as evidence (True, "merged", a prose blob).
            # Recorded as claimed-and-unsubstantiated so the refusal is specific.
            claim.setdefault("_unreadable_claim", value if isinstance(value, str) else True)
    for key in CLAIM_CARRIED_KEYS:
        if key in metadata:
            claim.setdefault(key, metadata[key])
    return claim


def pending_landing_children(conn, task_id: str) -> list[dict]:
    """Child cards that carry a landing contract and have not landed yet.

    The chain-head case: a parent must not close ahead of the leg that lands its
    work. Only *declared* contracts count — a review/QA child on ``local-only``
    never withholds its parent's completion, which is what keeps the ordinary
    review flow (parent completes, which releases the child) working.
    """
    rows = conn.execute(
        "SELECT t.id, t.title, t.status, t.completion_contract FROM task_links l "
        "JOIN tasks t ON t.id = l.child_id WHERE l.parent_id = ? ORDER BY t.id",
        (task_id,),
    ).fetchall()
    out: list[dict] = []
    for row in rows:
        status = _field(row, "status") or ""
        if status in {"done", "archived"}:
            continue
        if declares_landing_contract(row):
            out.append({
                "id": _field(row, "id"),
                "title": _field(row, "title"),
                "status": status,
                "completion_contract": _field(row, "completion_contract"),
            })
    return out


def landed_evidence(claim: Any) -> tuple[bool, list[str], dict]:
    """``(ok, missing, evidence)`` for a landing claim.

    ``missing`` names exactly which of the three accepted evidence forms the claim
    does not carry, so a refusal can be acted on without reading this module.
    """
    missing = [EVIDENCE_MERGED_COMMIT, EVIDENCE_DEPLOY_STAMP, EVIDENCE_LIVE_HASH]
    evidence: dict = {}
    if not isinstance(claim, dict):
        return False, missing, evidence
    merged = claim.get("merged_commit") or claim.get("merge_commit")
    if isinstance(merged, str) and _GIT_SHA_RE.match(merged.strip()):
        evidence["merged_commit"] = merged.strip()
    stamp = claim.get("deploy_stamp") or claim.get("deploy-stamp")
    stamp_commit = None
    if isinstance(stamp, str) and stamp.strip():
        stamp_commit = resolve_deploy_stamp(stamp)
        if stamp_commit:
            evidence["deploy_stamp"] = stamp.strip()
            evidence["deploy_stamp_commit"] = stamp_commit
    if stamp_commit and evidence.get("merged_commit") and stamp_commit != evidence["merged_commit"].lower():
        # The named target contradicts the claim it is offered as evidence for.
        return False, [EVIDENCE_STAMP_MISMATCH], {}
    live = claim.get("live_sha256")
    merged_blob = claim.get("merged_blob_sha256")
    if isinstance(live, str) and isinstance(merged_blob, str):
        live_hex, blob_hex = live.strip().lower(), merged_blob.strip().lower()
        if _SHA256_RE.match(live_hex) and _SHA256_RE.match(blob_hex) and live_hex == blob_hex:
            artifact = claim.get("artifact_path") or claim.get("live_path")
            if isinstance(artifact, str) and artifact.strip():
                on_disk = _file_sha256(artifact)
                if on_disk is not None and on_disk != live_hex:
                    # The artifact the claim names does not hash to the hash it
                    # carries: the claim is self-refuting, so a merge commit
                    # alongside it must not carry the card either.
                    return False, [EVIDENCE_LIVE_HASH_MISMATCH], {}
                if on_disk is not None:
                    evidence["artifact_path"] = artifact.strip()
                    evidence["artifact_sha256"] = on_disk
            evidence["live_sha256"] = live_hex
            evidence["merged_blob_sha256"] = blob_hex
        else:
            # A live hash that does not equal its own merged blob is a
            # contradiction, not partial evidence: the claim is self-refuting, so
            # a merge commit alongside it must not carry the card.
            return False, [EVIDENCE_LIVE_HASH_MISMATCH], {}
    return bool(evidence), ([] if evidence else missing), evidence


def landing_gap(metadata: Any, children: Optional[list[dict]] = None) -> Optional[dict]:
    """The single answer the completion gate needs.

    ``None`` when this card is not a landing card — it completes unchanged.
    Otherwise a dict naming ``source``, the ``missing`` evidence, the ``evidence``
    found and the ``children`` still withholding the landing. Callers with a
    connection pass :func:`pending_landing_children`; without one the child rule
    is skipped (the claim rule still applies).
    """
    claim = landing_claim(metadata)
    children = children or []
    if claim is not None:
        ok, missing, evidence = landed_evidence(claim)
        if ok:
            return None
        return {
            "source": "claim",
            "missing": missing,
            "evidence": evidence,
            "children": [],
            "unreadable_claim": claim.get("_unreadable_claim"),
        }
    if children:
        return {
            "source": "child",
            "missing": list((EVIDENCE_MERGED_COMMIT, EVIDENCE_DEPLOY_STAMP, EVIDENCE_LIVE_HASH)),
            "evidence": {},
            "children": children,
        }
    return None


def landing_evidence_record(metadata: Any) -> Optional[dict]:
    """The evidence a substantiated claim carries, for the card's own record.

    ``None`` when the completion claims no landing; a claim that names no
    accepted evidence never reaches here — the gate refuses it instead. What it
    returns is what the target actually said: for a deploy stamp, the commit read
    out of the stamp, so the card records the landing it was closed on rather
    than the claim it was offered.
    """
    claim = landing_claim(metadata)
    if claim is None:
        return None
    ok, _missing, evidence = landed_evidence(claim)
    if not ok:
        return None
    return {"source": "claim", "evidence": evidence}


def gap_message(task_id: str, gap: dict) -> str:
    """One actionable line naming the exact missing landed evidence."""
    if gap["source"] == "child":
        legs = ", ".join(f"{c['id']} ({c['status']}, {c['completion_contract']})" for c in gap["children"])
        return (
            f"completion blocked: {task_id} is a landing card whose own landing leg has not "
            f"landed: {legs}. A chain head must not close ahead of the leg that lands its work — "
            f"land the child (or archive it) first. Landed evidence is {EVIDENCE_MERGED_COMMIT}, "
            f"{EVIDENCE_DEPLOY_STAMP}, or {EVIDENCE_LIVE_HASH}."
        )
    detail = "; ".join(gap["missing"])
    if gap.get("unreadable_claim"):
        return (
            f"completion blocked: {task_id} claims a landing that names none of the accepted "
            f"evidence forms. Name {detail}."
        )
    return (
        f"completion blocked: {task_id} claims a landing that is not landed. Missing: {detail}. "
        f"A published or approved change is not a landing — being merged on the target branch is."
    )
