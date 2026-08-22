# Hermes Evidence Fabric v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add durable, profile-scoped ResearchRun, evidence, claim, provenance, and claim-evidence relationship primitives to Hermes without creating a second persistence system or modifying passed Desktop, Knowledge Hub, HUD, browser-isolation, or citation behavior.

**Architecture:** Extend the existing `SessionDB` SQLite schema and migration path with dedicated Evidence Fabric tables, while keeping DTOs, validation, ownership checks, and repository/service methods in a new `research` domain package. Reuse Hermes WAL, foreign keys, `BEGIN IMMEDIATE` retry behavior, and `hermes_home_key()` scope identity. Attempt non-core `research_fabric` exposure only after a trusted-runtime go/no-go proof; otherwise retain the service API only.

**Tech Stack:** Python 3.11+, SQLite, existing `SessionDB`, existing schema reconciliation/migration code, pytest, existing self-registering tool registry, and existing `toolsets.py` composition.

**Revision baseline:** This precision revision preserves prior plan commit `7f3e354064adfc910c4abb95b57bf45c823bd486`; it changes the plan only.

## Global Constraints

- Authoritative workspace: `C:\Users\curti\.hermes\hermes-agent`.
- Preserve all unrelated dirty worktree changes; stage only files named by the task being committed.
- Evidence Fabric is a distinct domain layer over shared `SessionDB`; it is not Hermes Memory, session history, or the existing optional `EvidenceStore` JSON script.
- No process-global authoritative research dictionaries, direct model SQL, filesystem database-path exposure, or arbitrary update API.
- Every new behavior follows failing-first TDD: write one focused test, run it and observe the expected failure, implement the smallest behavior, then rerun.
- `SCHEMA_VERSION` advances additively from current HEAD value `26`; no state database deletion, destructive unrelated migration, or new raw-page database is allowed.
- Run authority is the runtime-resolved `hermes_home_key()` scope; profile name and connection metadata are audit fields, not the sole authorization key.
- Terminal `ResearchRun` graphs are read-only; no v1 hard-delete or retention purge API is added.
- Evidence identity is enforced by SQLite partial unique indexes and service conflict handling, not by a service-only pre-check.
- Derived evidence uses a composite same-run foreign key; claims and links use composite same-run foreign keys.
- The service computes lowercase SHA-256 hashes from exact NFC UTF-8 text or exact binary bytes and performs conservative deterministic URI normalization.
- `source_type` and `retrieval_method` use separate constrained vocabularies with `OTHER` extension values.
- Claim status mutations and claim-evidence links record runtime-owned agent/profile provenance; model input cannot override it.
- Do not implement Query Graph, Adaptive Swarm Director, Skeptic, Verifier, reliability scoring, trust scoring, Knowledge Hub integration, Desktop UI, HUD, browser-isolation changes, Obsidian/NotebookLM integration, embeddings/vector storage, or local inference.

## File Map

- Create: `tests/test_evidence_fabric_schema.py` — schema, indexes, FKs, triggers, fresh/upgrade/idempotency tests.
- Create: `research/__init__.py` — public domain-package exports.
- Create: `research/evidence_fabric.py` — DTOs, enums, validation, URI/hash helpers, scope, repository/service methods, and errors.
- Create: `tests/test_evidence_fabric.py` — run, evidence, claim, provenance, validation, lifecycle, and scope tests.
- Create: `tests/test_evidence_fabric_concurrency.py` — real multi-connection writer and unique-index race tests.
- Conditional create: `tools/evidence_fabric_tool.py` — compact registered `research_fabric` tool only after Task 8 go/no-go passes.
- Create: `tests/tools/test_evidence_fabric_tool.py` — go/no-go result plus conditional registry, schema, toolset, and authorization tests.
- Modify: `hermes_state_common.py` — Evidence Fabric DDL/indexes/triggers and schema version 27.
- Modify: `hermes_state_schema.py` — additive v27 migration/reconciliation behavior.
- Conditional modify: `toolsets.py` — non-core `research` toolset containing only `research_fabric` after Task 8 approval.
- Do not modify: `tests/test_evidence_store.py`, `optional-skills/security/oss-forensics/scripts/evidence-store.py`, Desktop, Knowledge Hub, HUD, browser-isolation, or grounded-citation files.

## Public contracts shared by all implementation tasks

The worker must implement these exact names and types in
`research/evidence_fabric.py`; later tasks consume these contracts without
inventing alternate shapes.

```python
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping, Sequence

class ResearchRunStatus(StrEnum):
    OPEN = "OPEN"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

class ClaimStatus(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNRESOLVED = "UNRESOLVED"

class EvidenceRelation(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    CONTEXT = "CONTEXT"

SourceType = str  # validated against the source_type vocabulary in Task 3
RetrievalMethod = str  # validated against the retrieval_method vocabulary in Task 3

@dataclass(frozen=True)
class EvidenceScope:
    scope_key: str
    profile_name: str | None
    connection_id: str | None
    agent_id: str

    @classmethod
    def from_runtime(cls, *, agent_id: str) -> "EvidenceScope": ...

@dataclass(frozen=True)
class ResearchRun:
    id: str
    objective: str
    owner_scope_key: str
    owner_profile: str | None
    owner_connection_id: str | None
    status: ResearchRunStatus
    metadata: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    research_run_id: str
    source_type: SourceType
    retrieval_method: RetrievalMethod
    source_uri: str | None
    canonical_uri: str | None
    title: str | None
    publisher_or_origin: str | None
    published_at: datetime | None
    retrieved_at: datetime
    content_hash: str
    raw_reference: str | None
    relevant_passages: tuple[Mapping[str, Any], ...]
    created_by_agent: str
    created_by_profile: str | None
    provider: str | None
    model: str | None
    derived_from_evidence_id: str | None
    untrusted_external_content: bool
    metadata: Mapping[str, Any]
    created_at: datetime

@dataclass(frozen=True)
class ClaimRecord:
    id: str
    research_run_id: str
    text: str
    status: ClaimStatus
    created_by_agent: str
    created_by_profile: str | None
    updated_by_agent: str | None
    updated_by_profile: str | None
    metadata: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass(frozen=True)
class ClaimEvidenceLink:
    claim_id: str
    evidence_id: str
    research_run_id: str
    relation: EvidenceRelation
    passage_locator: Mapping[str, Any] | None
    created_by_agent: str
    created_by_profile: str | None
    created_at: datetime

@dataclass(frozen=True)
class EvidenceWriteResult:
    evidence: EvidenceRecord
    created: bool

class EvidenceFabricError(Exception):
    def __init__(self, message: str) -> None: ...
class EvidenceValidationError(EvidenceFabricError, ValueError): ...
class EvidenceNotFoundError(EvidenceFabricError, LookupError): ...
class EvidenceScopeError(EvidenceFabricError, PermissionError): ...
class EvidenceLifecycleError(EvidenceFabricError, ValueError): ...
class EvidenceIntegrityError(EvidenceFabricError): ...

def canonicalize_uri(uri: str) -> str: ...
def content_sha256(content: str | bytes) -> str: ...

class EvidenceFabricService:
    def __init__(self, db: "SessionDB", scope: EvidenceScope) -> None: ...
    def create_research_run(
        self, objective: str, *, metadata: Mapping[str, Any] | None = None,
        owner_connection_id: str | None = None,
    ) -> ResearchRun: ...
    def get_research_run(self, run_id: str) -> ResearchRun: ...
    def list_research_runs(self) -> tuple[ResearchRun, ...]: ...
    def transition_research_run(
        self, run_id: str, status: ResearchRunStatus,
    ) -> ResearchRun: ...
    def add_evidence(
        self, run_id: str, *, source_type: SourceType,
        retrieval_method: RetrievalMethod, content: str | bytes,
        expected_content_hash: str | None = None,
        source_uri: str | None = None, title: str | None = None,
        publisher_or_origin: str | None = None,
        published_at: datetime | None = None,
        retrieved_at: datetime | None = None,
        raw_reference: str | None = None,
        relevant_passages: Sequence[Mapping[str, Any]] = (),
        derived_from_evidence_id: str | None = None,
        untrusted_external_content: bool = True,
        metadata: Mapping[str, Any] | None = None,
        provider: str | None = None, model: str | None = None,
    ) -> EvidenceWriteResult: ...
    def get_evidence(self, evidence_id: str) -> EvidenceRecord: ...
    def list_evidence(self, run_id: str) -> tuple[EvidenceRecord, ...]: ...
    def create_claim(
        self, run_id: str, text: str, *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ClaimRecord: ...
    def get_claim(self, claim_id: str) -> ClaimRecord: ...
    def list_claims(self, run_id: str) -> tuple[ClaimRecord, ...]: ...
    def link_evidence_to_claim(
        self, claim_id: str, evidence_id: str, relation: EvidenceRelation,
        *, passage_locator: Mapping[str, Any] | None = None,
    ) -> ClaimEvidenceLink: ...
    def set_claim_status(
        self, claim_id: str, status: ClaimStatus,
    ) -> ClaimRecord: ...
```

`expected_content_hash` is verification-only: the service always computes the
authoritative hash from `content`; when the optional value is present it must
be lowercase 64-character SHA-256 matching the computed value, otherwise
`EvidenceValidationError` is raised. The service never trusts or stores an
unverified caller hash. `EvidenceWriteResult.created` is `True` for the
winning insert and `False` for an exact duplicate; both outcomes carry the
same `EvidenceRecord.id`. No duplicate-retrieval audit subsystem is added.

## Task 1: Define the schema contract with failing tests

**Files:** Create `tests/test_evidence_fabric_schema.py`.

**Interfaces:** Produces executable tests for the four tables, required columns, v27 migration, partial unique indexes, composite same-run FKs, and terminal graph rules.

- [ ] **Step 1: Write tests for fresh schema objects.** Construct `SessionDB(tmp_path / "state.db")`; assert tables `research_runs`, `evidence_records`, `claims`, and `claim_evidence_links`, indexes `ux_evidence_exact_uri_hash` and `ux_evidence_exact_raw_hash`, v27, and `PRAGMA foreign_keys=1`.
- [ ] **Step 2: Write tests for the v26-to-v27 upgrade and repeated startup.** Build a minimal v26 database with representative existing rows, open it through `SessionDB`, reopen it repeatedly, and assert existing rows remain unchanged and Evidence Fabric objects are created exactly once.
- [ ] **Step 3: Write the direct-SQL cross-run test.** Insert run A/B and evidence A/B with `sqlite3.connect()`, then execute `connection.execute("PRAGMA foreign_keys = ON")` before attempting an evidence row in A whose `derived_from_evidence_id` is evidence B; assert `sqlite3.IntegrityError`. This bypasses service validation while explicitly exercising SQLite FK enforcement.
- [ ] **Step 4: Write direct-SQL terminal lifecycle tests.** Execute `connection.execute("PRAGMA foreign_keys = ON")` on the raw connection before attempting `UPDATE research_runs SET status = 'OPEN'` and `UPDATE research_runs SET status = 'FAILED'` for a `COMPLETED` run; assert both raise `sqlite3.IntegrityError`. Also attempt evidence/claim/link inserts and claim-status updates after completion; assert database rejection.
- [ ] **Step 5: Run the red tests.** `python -m pytest tests/test_evidence_fabric_schema.py -q`. Expected failure: the new tables/indexes do not exist on HEAD.
- [ ] **Step 6: Commit the red tests.** `git add tests/test_evidence_fabric_schema.py; git commit -m "test: define Evidence Fabric schema contract"`.

## Task 2: Add the additive v27 SQLite schema and migration

**Files:** Modify `hermes_state_common.py` at `SCHEMA_VERSION`/`SCHEMA_SQL`; modify `hermes_state_schema.py` in the current reconciliation/migration path; test `tests/test_evidence_fabric_schema.py`.

**Interfaces:** Consumes the existing `SCHEMA_SQL`, `SCHEMA_VERSION`, `SessionSchemaMixin`, and `SessionDB` connection setup. Produces four durable tables and DB-level integrity/immutability enforcement.

- [ ] **Step 1: Implement tables with explicit ownership/audit fields, UTC timestamps, metadata, and bounded references.** Keep raw source bodies out of ordinary rows; store only passages and reconstructable `raw_reference` values.
- [ ] **Step 2: Add composite keys/FKs.** Give EvidenceRecord a unique `(id, research_run_id)` key; make derived evidence reference `(derived_from_evidence_id, research_run_id)`; make claim/link rows reference the same run through composite FKs.
- [ ] **Step 3: Add DB-enforced exact identity.** Add:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS ux_evidence_exact_uri_hash
ON evidence_records(research_run_id, canonical_uri, content_hash)
WHERE canonical_uri IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_evidence_exact_raw_hash
ON evidence_records(research_run_id, raw_reference, content_hash)
WHERE canonical_uri IS NULL AND raw_reference IS NOT NULL;
```

- [ ] **Step 4: Add terminal-run write protection.** Add triggers that reject graph inserts/updates when the parent run status is not `OPEN`, and reject every terminal-to-any-other-status update, including `COMPLETED -> OPEN` and `COMPLETED -> FAILED`; terminal status itself remains immutable.
- [ ] **Step 5: Wire the additive v27 migration.** Follow the current `SessionSchemaMixin` and `SCHEMA_SQL` source-of-truth conventions; bump 26 to 27; do not rewrite sessions, messages, memory, delegation, or FTS data.
- [ ] **Step 6: Run green schema tests.** `python -m pytest tests/test_evidence_fabric_schema.py -q`.
- [ ] **Step 7: Commit.** `git add hermes_state_common.py hermes_state_schema.py tests/test_evidence_fabric_schema.py; git commit -m "feat: add Evidence Fabric state schema"`.

## Task 3: Add domain DTOs, enums, scope, and deterministic validation

**Files:** Create `research/__init__.py`, `research/evidence_fabric.py`, and `tests/test_evidence_fabric.py`.

**Interfaces:** Produces `EvidenceScope`, `ResearchRunStatus`, `ClaimStatus`, `EvidenceRelation`, separate source/retrieval vocabularies, typed record DTOs, and controlled validation/authorization exceptions. `source_type` values are `WEB_SEARCH`, `WEB_PAGE`, `DOCUMENT`, `FILE`, `NOTE`, `NOTEBOOKLM`, `OBSIDIAN`, `HERMES_MEMORY`, and `OTHER`; `retrieval_method` values are `DIRECT_HTTP`, `BROWSER`, `BROWSER_VISION`, `API`, `MCP`, `FILE_READ`, `NOTEBOOKLM`, `OBSIDIAN`, `HERMES_INTERNAL`, and `OTHER`.

- [ ] **Step 1: Write failing validation tests.** Cover empty objectives/claim text, malformed IDs, invalid timestamps/enums, malformed hashes, invalid absolute URIs/raw references, and model-supplied mutation identity fields.
- [ ] **Step 2: Write failing deterministic helper tests.** Assert service-owned SHA-256 of NFC text and exact bytes; whitespace/newline changes remain changes; `expected_content_hash=None` is accepted; a matching expected hash is accepted; a malformed or mismatched expected hash raises `EvidenceValidationError`; URI normalization lowercases scheme/hostname, removes fragments/default ports, supplies `/`, and preserves query order/duplicates/casing/tracking parameters.

```python
def test_uri_normalization_is_conservative():
    assert canonicalize_uri("HTTPS://Example.COM:443") == "https://example.com/"
    assert canonicalize_uri("https://example.com/a?b=2&a=1#frag") == "https://example.com/a?b=2&a=1"
```

- [ ] **Step 3: Implement pure helpers and frozen DTOs.** `EvidenceScope.from_runtime(agent_id: str) -> EvidenceScope` derives `scope_key` from `hermes_home_key()` and carries runtime profile, connection, and agent identity. `content_sha256(content: str | bytes) -> str` computes the authoritative hash; `expected_content_hash` is only verification input.
- [ ] **Step 4: Run and commit.** `python -m pytest tests/test_evidence_fabric.py -q`; then `git add research tests/test_evidence_fabric.py; git commit -m "feat: add Evidence Fabric domain validation"`.

## Task 4: Implement ResearchRun repository/service behavior

**Files:** Modify `research/evidence_fabric.py`; test `tests/test_evidence_fabric.py`.

**Interfaces:** Produces the exact `EvidenceFabricService` methods in the Public contracts section. `create_research_run` returns `ResearchRun`; reads return `ResearchRun` or an immutable tuple; transitions return `ResearchRun` and raise `EvidenceLifecycleError` for invalid transitions or `EvidenceScopeError` for another scope.

- [ ] **Step 1: Write failing tests for create/retrieve/list, objective/owner retention, different-scope isolation, valid `OPEN` to terminal transitions, invalid reopen/terminal transitions, and graph-write rejection after terminal status.**
- [ ] **Step 2: Implement run writes/reads.** Check ownership and status inside the same `BEGIN IMMEDIATE` transaction; persist `owner_scope_key`, profile, connection ID, agent, metadata, and timestamps; return DTOs, never raw rows. The transition contract is:

```python
def transition_research_run(
    self, run_id: str, status: ResearchRunStatus,
) -> ResearchRun:
    # OPEN -> one terminal state; terminal -> anything raises EvidenceLifecycleError.
    ...
```

`get_research_run` and every mutation must query by both `id` and `scope_key`; a missing or foreign row raises `EvidenceNotFoundError`/`EvidenceScopeError` without revealing the other scope’s data.
- [ ] **Step 3: Run and commit.** `python -m pytest tests/test_evidence_fabric.py -q`; commit `feat: add scoped ResearchRun service`.

## Task 5: Implement evidence ingestion, provenance, deduplication, and versioning

**Files:** Modify `research/evidence_fabric.py`; test `tests/test_evidence_fabric.py`.

**Interfaces:** Produces the exact `add_evidence`, `get_evidence`, and `list_evidence` methods in the Public contracts section. `add_evidence` returns `EvidenceWriteResult`; reads return `EvidenceRecord` or an immutable tuple; malformed input raises `EvidenceValidationError`, missing IDs raise `EvidenceNotFoundError`, and cross-scope/run or terminal writes raise `EvidenceScopeError`/`EvidenceLifecycleError`.

- [ ] **Step 1: Write failing evidence tests.** Cover separate source/retrieval vocabularies, URI/timestamp/hash retention, original raw reference, derived parent chain, untrusted external text as data, same URI/hash duplicates, changed hashes as new versions, URI-less raw-reference identity, and terminal-run rejection.
- [ ] **Step 2: Implement service-owned ingestion.** Require source content bytes/text for hashing; compute `content_sha256(content)`; accept `expected_content_hash` only when it matches; normalize URI conservatively; execute authorization/status/insert in one transaction. The public call shape is:

```python
result = service.add_evidence(
    run_id,
    source_type="WEB_PAGE",
    retrieval_method="BROWSER",
    content=page_text,
    expected_content_hash=None,
    source_uri="https://example.com/spec#section-1",
    raw_reference="artifact://sha256/…",
)
assert result.evidence.content_hash == content_sha256(page_text)
```
- [ ] **Step 3: Handle unique-index races deterministically.** Catch the SQLite uniqueness conflict, re-read the exact identity, and return the committed evidence ID/duplicate result. Never expose raw `IntegrityError` for a legitimate identical concurrent insert and never overwrite an older source version.
- [ ] **Step 4: Run and commit.** `python -m pytest tests/test_evidence_fabric.py -q`; commit `feat: add durable EvidenceRecord service`.

## Task 6: Implement claims, explicit relationships, and mutation provenance

**Files:** Modify `research/evidence_fabric.py`; test `tests/test_evidence_fabric.py`.

**Interfaces:** Produces the exact claim/link methods in the Public contracts section. `create_claim`/`set_claim_status` return `ClaimRecord`, reads return `ClaimRecord` or an immutable tuple, and linking returns `ClaimEvidenceLink`; invalid IDs/relations/statuses raise the named domain errors rather than raw SQLite errors.

- [ ] **Step 1: Write failing tests.** Cover default `UNVERIFIED`, SUPPORTS/CONTRADICTS/CONTEXT, multiple links, simultaneous support/contradiction, no automatic status inference, invalid relations, nonexistent IDs, cross-run/cross-scope references, terminal rejection, and deterministic status transitions.
- [ ] **Step 2: Implement transactional checks and writes.** Check IDs, same run, ownership, and terminal status within the write transaction. Populate link `created_by_agent/profile` and claim `updated_by_agent/profile` from `EvidenceScope`; reject or ignore model-supplied identity fields. The relationship call must have this shape:

```python
link = service.link_evidence_to_claim(
    claim_id, evidence_id, EvidenceRelation.SUPPORTS,
    passage_locator={"quote_id": "p1"},
)
assert link.created_by_agent == scope.agent_id
```
- [ ] **Step 3: Run and commit.** `python -m pytest tests/test_evidence_fabric.py -q`; commit `feat: add claim ledger relationships`.

## Task 7: Prove concurrency, integrity, persistence, and migration end-to-end

**Files:** Create `tests/test_evidence_fabric_concurrency.py`; extend `tests/test_evidence_fabric.py` and `tests/test_evidence_fabric_schema.py`.

**Interfaces:** Produces regression proof for parallel agents, restart durability, direct database constraints, and rollback safety.

- [ ] **Step 1: Write the identical-evidence race test.** Use a barrier and two threads, each with its own `SessionDB` against one temporary DB, calling `add_evidence` with identical run/URI/content. Assert one row, equal `result.evidence.id` values, exactly one `result.created is True` and one `result.created is False` (allowing either caller to win), no raw `IntegrityError`, and distinct evidence from both writers preserved. Do not add a duplicate-retrieval audit table or event subsystem.
- [ ] **Step 2: Write concurrent distinct claim/link tests.** Assert no lost records, unique IDs, correct relations, and that one forced invalid transaction does not corrupt or remove successful records.
- [ ] **Step 3: Write close/reopen and upgrade tests.** Assert all records, hashes, timestamps, scope, and provenance survive restart; assert v26 upgrade is idempotent and preserves existing rows.
- [ ] **Step 4: Implement only minimum duplicate/retry handling.** Reuse existing WAL/`BEGIN IMMEDIATE` retry behavior; add no second lock, cache, or database.
- [ ] **Step 5: Run.** `python -m pytest tests/test_evidence_fabric_concurrency.py tests/test_evidence_fabric.py tests/test_evidence_fabric_schema.py -q`.
- [ ] **Step 6: Commit.** `git add tests/test_evidence_fabric_concurrency.py tests/test_evidence_fabric.py tests/test_evidence_fabric_schema.py; git commit -m "test: prove Evidence Fabric concurrency and durability"`.

## Task 8: Decide whether compact model-facing exposure is safe

**Files:** Create `tests/tools/test_evidence_fabric_tool.py`; create `tools/evidence_fabric_tool.py` and modify `toolsets.py` only after the go decision is green.

**Interfaces:** Consumes `EvidenceFabricService` and `EvidenceScope.from_runtime`. Produces either the registered `research_fabric` tool in a non-core `research` toolset or the explicit test/report marker `MODEL_TOOL_DEFERRED`; never a tool that accepts model-controlled scope/run authority.

- [ ] **Step 1: Write the go/no-go test and runtime proof.** Inspect and test the actual trusted tool-dispatch context available to `tools/evidence_fabric_tool.py`: it must derive `EvidenceScope.from_runtime(agent_id=...)` and an authorized `run_id` from trusted runtime context, not from a model parameter. The test must prove that a model-supplied `scope_key`, profile, connection, or alternate run authority is ignored/rejected. If no trusted run context exists in current Hermes dispatch, the test records `MODEL_TOOL_DEFERRED` and the service-only API remains the supported surface.
- [ ] **Step 2: If and only if the proof is green, write tool-contract tests.** Assert discovery, the exact compact operation schema, absence from `_HERMES_CORE_TOOLS`, opt-in `research` toolset, malformed-operation rejection, active `hermes_home_key()` use, and no SQL/database-path/scope-authority parameters. If the proof is not green, assert the defer marker and do not create production tool files.
- [ ] **Step 3: If and only if approved by Step 1, implement the registered tool.** Follow `tools/registry.py`; derive trusted runtime identity in the handler; pass `EvidenceScope` and trusted run context to the service; dispatch only the Public-contract methods; bound JSON and return controlled domain errors. Otherwise document `MODEL_TOOL_DEFERRED` and skip `tools/evidence_fabric_tool.py` and `toolsets.py` changes.
- [ ] **Step 4: Run the branch-specific tests and commit only the selected outcome.** Tool branch: `python -m pytest tests/tools/test_evidence_fabric_tool.py tests/test_evidence_fabric.py -q` and commit `feat: expose compact Evidence Fabric tool`; deferred branch: run the service suite, retain only the test/report marker, and commit `docs/test: defer Evidence Fabric model tool`.

## Task 9: Run compatibility and regression verification

**Files:** Existing tests only; modify production code only for a concrete failing regression, with a regression test first.

- [ ] **Step 1: Run state/delegation regressions.**

```powershell
python -m pytest tests/test_hermes_state.py tests/test_hermes_state_wal_fallback.py tests/test_session_db_context_manager.py tests/test_session_db_read_conn_pool.py tests/test_delegate_cascade_49148.py tests/tools/test_delegate.py tests/tools/test_delegate_toolset_scope.py -q
```

- [ ] **Step 2: Run grounded-citation compatibility unchanged.** `python -m pytest tests/skills/test_grounded_citations_skill.py -q`; verify no citation files changed and no new code treats its JSON ledger as Evidence Fabric authority.
- [ ] **Step 3: Run changed-module compile and focused tests.**

```powershell
python -m compileall hermes_state_common.py hermes_state_schema.py research
python -m pytest tests/test_evidence_fabric_schema.py tests/test_evidence_fabric.py tests/test_evidence_fabric_concurrency.py tests/tools/test_evidence_fabric_tool.py -q
```

- [ ] **Step 4: Commit only concrete fixes.** Use a separate red-green commit for each actual regression; do not reopen passed boundaries for cleanup.

## Task 10: Adversarial self-check and final verification

**Files:** Inspect all changed files and final Git diff; create no new production files.

- [ ] **Step 1: Search for forbidden patterns.** Search changed code for process-global evidence stores, SQL/database-path tool inputs, joins missing `research_run_id`, profile-name-only authority, overwritten source versions, provenance-free summaries, fake confidence/trust, raw page bodies in ordinary rows, local-model code, or Knowledge Hub/Desktop/HUD/browser edits. Review every match manually.
- [ ] **Step 2: Inspect scope and whitespace.** Run `git status --short --branch`, `git diff --check`, and a final diff limited to the File Map. Confirm unrelated dirty changes remain untouched.
- [ ] **Step 3: Run the canonical targeted runner.** `scripts/run_tests.sh tests/test_evidence_fabric_schema.py tests/test_evidence_fabric.py tests/test_evidence_fabric_concurrency.py tests/tools/test_evidence_fabric_tool.py`; then run Task 9’s regression commands. Run the full `scripts/run_tests.sh` only if the bounded environment permits it and report any skipped scope explicitly.
- [ ] **Step 4: Produce the final evidence-backed report.** Include workspace, branch, initial/final HEAD, changed files, migration/persistence/concurrency/integrity/scope/provenance results, exact commands and exit results, regressions, adversarial findings, and deferred layers. Use `EVIDENCE_FABRIC_V1_READY` only with fresh passing evidence; otherwise report the specific blocker.

## Plan self-review checklist

- [ ] The six approved hardening requirements are covered: DB partial unique indexes, composite derived FK, terminal read-only graph, deterministic service hashing, conservative URI normalization, separate vocabularies, and runtime-owned mutation provenance.
- [ ] The unique-index race and direct-SQL cross-run derivation tests are explicit.
- [ ] Schema/migration precedes DTO/service work, which precedes tool exposure.
- [ ] Every new production module has a named failing test file and focused command.
- [ ] No passed boundary or unrelated dirty file is in scope.
- [ ] No placeholder, vague behavior, unresolved architecture choice, or accidental future-layer work remains.
