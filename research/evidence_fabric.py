"""Evidence Fabric domain services over Hermes' durable SessionDB."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from hermes_constants import hermes_home_key

SOURCE_TYPES = frozenset({"WEB_SEARCH", "WEB_PAGE", "DOCUMENT", "FILE", "NOTE", "NOTEBOOKLM", "OBSIDIAN", "HERMES_MEMORY", "OTHER"})
RETRIEVAL_METHODS = frozenset({"DIRECT_HTTP", "BROWSER", "BROWSER_VISION", "API", "MCP", "FILE_READ", "NOTEBOOKLM", "OBSIDIAN", "HERMES_INTERNAL", "OTHER"})

class ResearchRunStatus(StrEnum):
    OPEN = "OPEN"; COMPLETED = "COMPLETED"; CANCELLED = "CANCELLED"; FAILED = "FAILED"
class ClaimStatus(StrEnum):
    UNVERIFIED = "UNVERIFIED"; SUPPORTED = "SUPPORTED"; PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"; CONTRADICTED = "CONTRADICTED"; UNRESOLVED = "UNRESOLVED"
class EvidenceRelation(StrEnum):
    SUPPORTS = "SUPPORTS"; CONTRADICTS = "CONTRADICTS"; CONTEXT = "CONTEXT"
SourceType = str
RetrievalMethod = str

@dataclass(frozen=True)
class EvidenceScope:
    scope_key: str; profile_name: str | None; connection_id: str | None; agent_id: str
    @classmethod
    def from_runtime(cls, *, agent_id: str) -> "EvidenceScope":
        return cls(hermes_home_key(), None, None, agent_id)

@dataclass(frozen=True)
class ResearchRun:
    id: str; objective: str; owner_scope_key: str; owner_profile: str | None; owner_connection_id: str | None; status: ResearchRunStatus; metadata: Mapping[str, Any]; created_at: datetime; updated_at: datetime
@dataclass(frozen=True)
class EvidenceRecord:
    id: str; research_run_id: str; source_type: SourceType; retrieval_method: RetrievalMethod; source_uri: str | None; canonical_uri: str | None; title: str | None; publisher_or_origin: str | None; published_at: datetime | None; retrieved_at: datetime; content_hash: str; raw_reference: str | None; relevant_passages: tuple[Mapping[str, Any], ...]; created_by_agent: str; created_by_profile: str | None; provider: str | None; model: str | None; derived_from_evidence_id: str | None; untrusted_external_content: bool; metadata: Mapping[str, Any]; created_at: datetime
@dataclass(frozen=True)
class ClaimRecord:
    id: str; research_run_id: str; text: str; status: ClaimStatus; created_by_agent: str; created_by_profile: str | None; updated_by_agent: str | None; updated_by_profile: str | None; metadata: Mapping[str, Any]; created_at: datetime; updated_at: datetime
@dataclass(frozen=True)
class ClaimEvidenceLink:
    claim_id: str; evidence_id: str; research_run_id: str; relation: EvidenceRelation; passage_locator: Mapping[str, Any] | None; created_by_agent: str; created_by_profile: str | None; created_at: datetime
@dataclass(frozen=True)
class EvidenceWriteResult:
    evidence: EvidenceRecord; created: bool

class EvidenceFabricError(Exception): pass
class EvidenceValidationError(EvidenceFabricError, ValueError): pass
class EvidenceNotFoundError(EvidenceFabricError, LookupError): pass
class EvidenceScopeError(EvidenceFabricError, PermissionError): pass
class EvidenceLifecycleError(EvidenceFabricError, ValueError): pass
class EvidenceIntegrityError(EvidenceFabricError): pass

def _now() -> datetime: return datetime.now(timezone.utc)
def _epoch(value: datetime | None) -> float | None: return value.timestamp() if value else None
def _dt(value: float | int | None) -> datetime | None: return datetime.fromtimestamp(value, timezone.utc) if value is not None else None
def _json(value: Any) -> str: return json.dumps({} if value is None else value, sort_keys=True, separators=(",", ":"))
def _id(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 200: raise EvidenceValidationError("invalid identifier")
    return value

def content_sha256(content: str | bytes) -> str:
    if isinstance(content, str): content = unicodedata.normalize("NFC", content).encode("utf-8")
    elif not isinstance(content, bytes): raise EvidenceValidationError("content must be str or bytes")
    return hashlib.sha256(content).hexdigest()

def canonicalize_uri(uri: str) -> str:
    if not isinstance(uri, str) or not uri.strip(): raise EvidenceValidationError("URI is required")
    parts = urlsplit(uri.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc: raise EvidenceValidationError("URI must be absolute HTTP(S)")
    host = parts.hostname.lower() if parts.hostname else ""
    if not host: raise EvidenceValidationError("URI host is required")
    try: port = parts.port
    except ValueError as exc: raise EvidenceValidationError("invalid URI port") from exc
    default = (parts.scheme.lower() == "http" and port == 80) or (parts.scheme.lower() == "https" and port == 443)
    netloc = host if not port or default else f"{host}:{port}"
    if parts.username or parts.password: raise EvidenceValidationError("URI credentials are not allowed")
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), netloc, path, parts.query, ""))

class EvidenceFabricService:
    def __init__(self, db: "SessionDB", scope: EvidenceScope) -> None: self.db, self.scope = db, scope
    def _write(self, fn): return self.db._execute_write(fn)
    def _fetch(self, sql, params=()):
        with self.db._lock: return self.db._conn.execute(sql, params).fetchall()
    def _run(self, run_id):
        rows = self._fetch("SELECT * FROM research_runs WHERE id=?", (run_id,))
        if not rows: raise EvidenceNotFoundError("research run not found")
        row = rows[0]
        if row["owner_scope_key"] != self.scope.scope_key: raise EvidenceScopeError("run belongs to another scope")
        return row
    def _require_open_run(self, run_id: str):
        row = self._run(_id(run_id))
        if row["status"] != ResearchRunStatus.OPEN.value:
            raise EvidenceLifecycleError("terminal research run is immutable")
        return row
    @staticmethod
    def _is_lifecycle_integrity_error(exc: sqlite3.IntegrityError) -> bool:
        return str(exc) in {"research run is not open", "terminal research run is immutable"}
    def _run_dto(self, r): return ResearchRun(r["id"],r["objective"],r["owner_scope_key"],r["owner_profile"],r["owner_connection_id"],ResearchRunStatus(r["status"]),json.loads(r["metadata_json"]),_dt(r["created_at"]),_dt(r["updated_at"]))
    def create_research_run(self, objective: str, *, metadata: Mapping[str, Any] | None = None, owner_connection_id: str | None = None) -> ResearchRun:
        if not isinstance(objective, str) or not objective.strip(): raise EvidenceValidationError("objective is required")
        now = _now(); rid = str(uuid.uuid4())
        def write(c): c.execute("INSERT INTO research_runs VALUES (?,?,?,?,?,?,?,?,?)", (rid, objective, self.scope.scope_key, self.scope.profile_name, owner_connection_id or self.scope.connection_id, "OPEN", _json(metadata), now.timestamp(), now.timestamp()))
        self._write(write); return self.get_research_run(rid)
    def get_research_run(self, run_id: str) -> ResearchRun: return self._run_dto(self._run(_id(run_id)))
    def list_research_runs(self) -> tuple[ResearchRun, ...]: return tuple(self._run_dto(r) for r in self._fetch("SELECT * FROM research_runs WHERE owner_scope_key=? ORDER BY created_at", (self.scope.scope_key,)))
    def transition_research_run(self, run_id: str, status: ResearchRunStatus) -> ResearchRun:
        row = self._run(_id(run_id)); status = ResearchRunStatus(status)
        if row["status"] != "OPEN" or status.value == row["status"]: raise EvidenceLifecycleError("invalid research run transition")
        now = _now()
        try: self._write(lambda c: c.execute("UPDATE research_runs SET status=?,updated_at=? WHERE id=?", (status.value,now.timestamp(),run_id)))
        except sqlite3.IntegrityError as exc: raise EvidenceLifecycleError(str(exc)) from exc
        return self.get_research_run(run_id)
    def _evidence_dto(self, r): return EvidenceRecord(r["id"],r["research_run_id"],r["source_type"],r["retrieval_method"],r["source_uri"],r["canonical_uri"],r["title"],r["publisher_or_origin"],_dt(r["published_at"]),_dt(r["retrieved_at"]),r["content_hash"],r["raw_reference"],tuple(json.loads(r["relevant_passages_json"])),r["created_by_agent"],r["created_by_profile"],r["provider"],r["model"],r["derived_from_evidence_id"],bool(r["untrusted_external_content"]),json.loads(r["metadata_json"]),_dt(r["created_at"]))
    def add_evidence(self, run_id: str, *, source_type: SourceType, retrieval_method: RetrievalMethod, content: str | bytes, expected_content_hash: str | None = None, source_uri: str | None = None, title: str | None = None, publisher_or_origin: str | None = None, published_at: datetime | None = None, retrieved_at: datetime | None = None, raw_reference: str | None = None, relevant_passages: Sequence[Mapping[str, Any]] = (), derived_from_evidence_id: str | None = None, untrusted_external_content: bool = True, metadata: Mapping[str, Any] | None = None, provider: str | None = None, model: str | None = None) -> EvidenceWriteResult:
        self._require_open_run(run_id)
        if source_type not in SOURCE_TYPES or retrieval_method not in RETRIEVAL_METHODS: raise EvidenceValidationError("invalid source or retrieval vocabulary")
        digest = content_sha256(content)
        if expected_content_hash is not None and (expected_content_hash != digest or len(expected_content_hash) != 64 or expected_content_hash.lower() != expected_content_hash): raise EvidenceValidationError("expected_content_hash does not match content")
        canonical = canonicalize_uri(source_uri) if source_uri else None
        if not canonical and not raw_reference: raise EvidenceValidationError("source_uri or raw_reference is required")
        eid = str(uuid.uuid4()); now = _now(); retrieved = retrieved_at or now
        values=(eid,run_id,source_type,retrieval_method,source_uri,canonical,title,publisher_or_origin,_epoch(published_at),_epoch(retrieved),digest,raw_reference,_json(list(relevant_passages)),self.scope.agent_id,self.scope.profile_name,provider,model,derived_from_evidence_id,int(bool(untrusted_external_content)),_json(metadata),now.timestamp())
        try:
            self._write(lambda c: c.execute("INSERT INTO evidence_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)); return EvidenceWriteResult(self.get_evidence(eid), True)
        except sqlite3.IntegrityError as exc:
            if self._is_lifecycle_integrity_error(exc): raise EvidenceLifecycleError(str(exc)) from exc
            if "UNIQUE" not in str(exc): raise EvidenceIntegrityError(str(exc)) from exc
            rows=self._fetch("SELECT * FROM evidence_records WHERE research_run_id=? AND content_hash=? AND ((canonical_uri IS NOT NULL AND canonical_uri=?) OR (canonical_uri IS NULL AND raw_reference=?))", (run_id,digest,canonical,raw_reference))
            if not rows: raise EvidenceIntegrityError(str(exc)) from exc
            return EvidenceWriteResult(self._evidence_dto(rows[0]), False)
    def get_evidence(self, evidence_id: str) -> EvidenceRecord:
        rows=self._fetch("SELECT e.* FROM evidence_records e JOIN research_runs r ON r.id=e.research_run_id WHERE e.id=? AND r.owner_scope_key=?", (_id(evidence_id),self.scope.scope_key))
        if not rows: raise EvidenceNotFoundError("evidence not found")
        return self._evidence_dto(rows[0])
    def list_evidence(self, run_id: str) -> tuple[EvidenceRecord, ...]: self._run(run_id); return tuple(self._evidence_dto(r) for r in self._fetch("SELECT * FROM evidence_records WHERE research_run_id=? ORDER BY created_at", (run_id,)))
    def create_claim(self, run_id: str, text: str, *, metadata: Mapping[str, Any] | None = None) -> ClaimRecord:
        self._require_open_run(run_id)
        if not isinstance(text,str) or not text.strip(): raise EvidenceValidationError("claim text is required")
        cid=str(uuid.uuid4()); now=_now()
        try: self._write(lambda c:c.execute("INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?)",(cid,run_id,text,"UNVERIFIED",self.scope.agent_id,self.scope.profile_name,None,None,_json(metadata),now.timestamp(),now.timestamp())))
        except sqlite3.IntegrityError as exc:
            if self._is_lifecycle_integrity_error(exc): raise EvidenceLifecycleError(str(exc)) from exc
            raise EvidenceIntegrityError(str(exc)) from exc
        return self.get_claim(cid)
    def _claim_dto(self,r): return ClaimRecord(r["id"],r["research_run_id"],r["text"],ClaimStatus(r["status"]),r["created_by_agent"],r["created_by_profile"],r["updated_by_agent"],r["updated_by_profile"],json.loads(r["metadata_json"]),_dt(r["created_at"]),_dt(r["updated_at"]))
    def get_claim(self, claim_id: str) -> ClaimRecord:
        rows=self._fetch("SELECT c.* FROM claims c JOIN research_runs r ON r.id=c.research_run_id WHERE c.id=? AND r.owner_scope_key=?",(_id(claim_id),self.scope.scope_key))
        if not rows: raise EvidenceNotFoundError("claim not found")
        return self._claim_dto(rows[0])
    def list_claims(self, run_id: str) -> tuple[ClaimRecord, ...]: self._run(run_id); return tuple(self._claim_dto(r) for r in self._fetch("SELECT * FROM claims WHERE research_run_id=? ORDER BY created_at",(run_id,)))
    def link_evidence_to_claim(self, claim_id: str, evidence_id: str, relation: EvidenceRelation, *, passage_locator: Mapping[str, Any] | None = None) -> ClaimEvidenceLink:
        claim=self.get_claim(claim_id); evidence=self.get_evidence(evidence_id)
        if claim.research_run_id != evidence.research_run_id: raise EvidenceIntegrityError("claim and evidence belong to different runs")
        self._require_open_run(claim.research_run_id)
        now=_now(); relation=EvidenceRelation(relation)
        try:self._write(lambda c:c.execute("INSERT INTO claim_evidence_links VALUES (?,?,?,?,?,?,?,?)",(claim_id,evidence_id,claim.research_run_id,relation.value,_json(passage_locator) if passage_locator else None,self.scope.agent_id,self.scope.profile_name,now.timestamp())))
        except sqlite3.IntegrityError as exc:
            if self._is_lifecycle_integrity_error(exc): raise EvidenceLifecycleError(str(exc)) from exc
            raise EvidenceIntegrityError(str(exc)) from exc
        return ClaimEvidenceLink(claim_id,evidence_id,claim.research_run_id,relation,passage_locator,self.scope.agent_id,self.scope.profile_name,now)
    def set_claim_status(self, claim_id: str, status: ClaimStatus) -> ClaimRecord:
        claim = self.get_claim(claim_id); self._require_open_run(claim.research_run_id); status=ClaimStatus(status); now=_now()
        try:self._write(lambda c:c.execute("UPDATE claims SET status=?,updated_by_agent=?,updated_by_profile=?,updated_at=? WHERE id=?",(status.value,self.scope.agent_id,self.scope.profile_name,now.timestamp(),claim_id)))
        except sqlite3.IntegrityError as exc:
            if self._is_lifecycle_integrity_error(exc): raise EvidenceLifecycleError(str(exc)) from exc
            raise EvidenceIntegrityError(str(exc)) from exc
        return self.get_claim(claim_id)
