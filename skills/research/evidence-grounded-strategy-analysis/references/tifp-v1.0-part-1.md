# Transcript-to-Initial-Findings Protocol — v1.0

**Protocol ID:** TIFP  
**Version:** 1.0.0  
**Protocol owner role:** TIFP Protocol Owner  
**Protocol release authority:** TIFP Protocol Owner  
**Status:** Final release candidate; release is effective only when this exact hash is named in `protocol-development/RELEASE.json`  
**Normative terms:** Only numbered statements containing **SHALL** or **SHALL NOT** are requirements. IDs provide traceability, not a claim that every sentence is intrinsically atomic or self-testing. Conformance uses the requirement-trace metadata in Section 27.

## 1. Scope and non-goals

This protocol governs conversion of one or more current-engagement discovery transcripts into two separately controlled artifacts: a readable Initial Findings Report and an Evidence and Verification Dossier.

- **TIFP-SCP-001 — SHALL**: A run SHALL process evidence from exactly one engagement.
- **TIFP-SCP-002 — SHALL**: A run SHALL produce one client-facing Initial Findings Report.
- **TIFP-SCP-003 — SHALL**: A run SHALL produce one separately access-controlled Evidence and Verification Dossier.
- **TIFP-SCP-004 — SHALL**: The reusable protocol SHALL remain client-neutral.
- **TIFP-SCP-005 — SHALL**: Engagement evidence SHALL remain outside the reusable protocol.
- **TIFP-SCP-006 — SHALL**: The protocol SHALL NOT be treated as legal, regulatory, or records-retention advice.
- **TIFP-SCP-007 — SHALL**: The protocol SHALL NOT establish that meeting participants represent an entire organization.
- **TIFP-SCP-008 — SHALL**: The protocol SHALL NOT convert an initial finding into a final audit conclusion.
- **TIFP-SCP-009 — SHALL**: The protocol SHALL NOT select an implementation architecture, model, provider, or storage platform.
- **TIFP-SCP-010 — SHALL**: Every engagement run SHALL copy at least one complete primary transcript-text source during S1 and hash-pin it in the frozen source manifest at S2 before evidence extraction or analysis.

## 2. Defined artifact classes

| Class | Definition |
|---|---|
| Protocol | This hash-pinned, client-neutral specification. |
| Primary source | Complete transcript text supplied by the current engagement or canonical transcript text derived from a recording; a recording without transcript text is supplementary, not a qualifying primary transcript. |
| Derivative source | A summary, highlight, meeting note, excerpt collection, or transformation dependent on another source. |
| Current-engagement instruction | A run-specific instruction approved and manifest-listed for the current engagement. |
| Source manifest | The frozen inventory and integrity record for current-engagement evidence and instructions. |
| Isolation policy | The run-specific positive identifier policy and secondary prohibited-identifier list. |
| Stage-context manifest | Frozen canonical inventory authorizing context for exactly one stage; it references but never modifies the source manifest. |
| Reviewer-control manifest | Pre-bound stage-context manifest containing reviewer control instructions frozen before subject-set binding. |
| Bound generated artifact | Report, dossier, or validation artifact generated in this run, hash-pinned for S7 or S9, and always treated as untrusted data rather than instructions. |
| Outbound context manifest | The byte-addressable inventory of the exact payload sent to a model invocation. |
| Evidence ledger | Structured pre-synthesis extraction of claims and their source locations. |
| Quote ledger | Structured record used to validate every direct quotation. |
| Initial Findings Report | Natural-prose client artifact. |
| Evidence and Verification Dossier | Restricted audit artifact containing evidence, mappings, checks, and dispositions. |
| Subject-set manifest | Frozen binding of report, dossier, validation artifacts, source manifest, protocol, instructions, and reviewer controls reviewed together. |
| Review disposition | Immutable append-only control record outside the dossier bound to one subject-set manifest. |
| Publication record | Immutable append-only control record outside the dossier bound to a passing disposition and its subject set. |
| Advisory-acceptance record | Immutable append-only audit-control record bound to one advisory, disposition, subject set, and predecessor control hash. |
| Advisory-resolution record | Immutable append-only audit-control record documenting correction, withdrawal, supersession, or blocker reclassification of one recorded advisory. |
| Engagement-authorization record | Frozen current-engagement instruction establishing approved purposes, audiences, and release authority. |
| Model-response capture record | Canonical request/result lineage record for one model invocation. |
| Model-service-boundary record | Frozen pre-transmission record of the context-builder/provider interface and approved transmission/access controls. |
| Raw model-response bytes | Quarantined immutable provider-returned bytes that are never model context. |
| Raw-response manifest | Immutable inventory of quarantined raw-response paths, byte counts, hashes, capture records, and access policy. |
| Draft | Mutable pre-binding working artifact beneath `work/drafts/`. |
| Validation artifact | Immutable pre-G7 or G7 validation result authorized only as untrusted validation/review data at its declared stage. |
| Closure-attempt record | Immutable record of one G11 execution or retry. |
| Closure-exception record | Immutable historical open snapshot of one failed, errored, or incomplete G11 attempt. |
| Closure record | Immutable terminal audit-control record bound to the latest available run artifacts and prior control-chain head. |
| Requirements verification plan | Pre-S8 immutable audit-control record assigning every requirement an owner, verifier, method, source trace, and applicability trigger without asserting future results. |
| Requirements trace event | Immutable append-only audit-control record containing one requirement result and chaining to the prior event hash. |
| Acceptance-test result | Immutable observed-output record for one protocol acceptance test. |
| Acceptance-test manifest | Immutable inventory binding all implementation acceptance-test results to one protocol and implementation version. |
| Conformance claim | External presentation assertion about a frozen run audit package; it is not a run artifact, is not inserted into the package or trace, and is checked by the recipient or external verifier against cited immutable hashes. |
| Bound artifact set | Artifacts in one subject-set manifest. “Current” means generated in this run and bound in this run's current subject set only. |
| Material claim | A statement whose removal or correction could change a finding, recommendation, decision, risk, limitation, or reader interpretation. |
| Blocker | A failed non-waivable condition that prevents model invocation or publication, according to stage. |
| Advisory finding | A nonblocking style preference or disclosed methodological limitation that the protocol owner may accept with recorded rationale. |
| Recorded advisory | Every advisory ID present in the current bound review disposition or other current bound challenge/disposition record and not superseded by a blocker classification. |

**Operational definitions.** **Privacy leakage** is any other-engagement identity, fact, quotation, conclusion, metadata, attachment content, link target, or embedded object in the engagement root, evidence, context, generated artifact, or publication package, or current-engagement information exposed unless both the audience-role identifier and purpose identifier—and that exact audience/purpose pair—are expressly approved in the frozen engagement-authorization record. Detection is the union of positive provenance/path/class/hash authorization, package-region inspection, secondary identifier/pattern scans, and human semantic review; an inaccessible region is a blocker, not evidence of absence. **Reliable diarization** means authenticated speaker identity or designated-human verification against the recording for the cited span, with no unresolved conflict; automated labels or speaker-name text alone are unreliable. **Materially relevant surrounding context** is the minimum contiguous span preserving referents, negation, qualification, conditions, uncertainty, speech act, speaker turn, and contradiction that could change a reasonable interpretation. **Outside-view treatment** means either comparison with an identified independent reference class and source/location, or an explicit statement that none was available plus sensitivity/range treatment and the resulting limitation. **Independent reviewer** means a separately assigned review role in a fresh isolated invocation with no drafting conversation, memory, or editing authority; it need not use a different underlying model, and any model overlap must be disclosed in the immutable review disposition.

- **TIFP-DEF-001 — SHALL**: Each artifact SHALL declare exactly one artifact class.
- **TIFP-DEF-002 — SHALL**: Each artifact SHALL carry a unique artifact identifier.
- **TIFP-DEF-003 — SHALL**: Each immutable artifact SHALL carry its SHA-256 hash.
- **TIFP-DEF-004 — SHALL**: Each artifact timestamp SHALL use ISO 8601 with an explicit time-zone designator.
- **TIFP-DEF-005 — SHALL**: `artifact_class` SHALL be the snake-case token of exactly one class listed in Section 2.
- **TIFP-DEF-006 — SHALL**: An unlisted or compound artifact class SHALL fail artifact validation.

## 3. Mandatory engagement workspace layout

A newly created workspace uses this relative layout; an implementation chooses and records the absolute engagement root.

```text
<engagement-root>/
  isolation-policy.json
  source-manifest.json
  instructions/
    engagement-authorization.json
    engagement-authorization-payload.json
    approvals/
      authorization-record-approval.json
  evidence/
    primary/
    derivative/
  work/
    stage-context-manifests/
    context-manifests/
    responses/
      raw/
      raw-response-manifest.json
      raw-response-access.jsonl
    ledgers/
    drafts/
  deliverables/
    initial-findings-report.*
    evidence-verification-dossier.*
  review/
    reviewer-controls/
    subject-sets/
    dispositions/
    advisory-acceptances/
    advisory-resolutions/
    requirements-trace/
    validation-results/
  publication/
    decisions/
  closure/
    attempts/
    exceptions/
```

- **TIFP-WRK-001 — SHALL**: Each engagement SHALL begin in a newly created engagement-specific root.
- **TIFP-WRK-002 — SHALL**: The engagement root SHALL be recorded as a canonical absolute path.
- **TIFP-WRK-003 — SHALL**: Current-engagement evidence SHALL reside beneath `<engagement-root>/evidence/`.
- **TIFP-WRK-004 — SHALL**: Current-engagement instructions SHALL reside beneath `<engagement-root>/instructions/`.
- **TIFP-WRK-005 — SHALL**: Generated work artifacts SHALL reside beneath `<engagement-root>/work/`.
- **TIFP-WRK-006 — SHALL**: Deliverable candidates SHALL reside beneath `<engagement-root>/deliverables/`.
- **TIFP-WRK-007 — SHALL**: Review records SHALL reside beneath `<engagement-root>/review/` and outside the dossier.
- **TIFP-WRK-007A — SHALL**: Review dispositions SHALL be immutable append-only records beneath `review/dispositions/`.
- **TIFP-WRK-007B — SHALL**: Publication decisions SHALL be immutable append-only records beneath `publication/decisions/`.
- **TIFP-WRK-007C — SHALL**: Advisory acceptances SHALL be immutable append-only records beneath `review/advisory-acceptances/`.
- **TIFP-WRK-007F — SHALL**: Advisory resolutions SHALL be immutable append-only records beneath `review/advisory-resolutions/`.
- **TIFP-WRK-007D — SHALL**: Requirements verification plans and trace events SHALL reside beneath `review/requirements-trace/`.
- **TIFP-WRK-007E — SHALL**: Closure records SHALL be immutable terminal records beneath `closure/`.
- **TIFP-WRK-008 — SHALL**: Only report/dossier distribution copies and publication payloads that passed the publication gate SHALL enter `<engagement-root>/publication/` outside `publication/decisions/`.
- **TIFP-WRK-008A — SHALL**: Publication-decision records MAY be created beneath `publication/decisions/` to decide the gate but SHALL remain immutable control records and SHALL NOT be treated as published deliverables.
- **TIFP-WRK-009 — SHALL**: Path authorization SHALL use canonical paths after resolving symlinks.
- **TIFP-WRK-010 — SHALL**: A symlink resolving outside the engagement root SHALL be rejected.
- **TIFP-WRK-011 — SHALL**: The reusable protocol SHALL be stored outside engagement evidence.

The pre-approval payload at `instructions/engagement-authorization-payload.json` contains `format` (`tifp-engagement-authorization-payload-v1`), engagement/run IDs, approved purpose identifiers/descriptions, approved audience-role identifiers, approved audience/purpose pairs, `approved_model_services[]` entries (provider/model, exact destination, `training_use` false, maximum provider retention, authorized provider-side roles, approved regions, and minimum transport standard), engagement owner, privacy approver, publication authority, expiration or explicit no-expiration rationale, and `payload_sha256`. Canonical approval evidence at `instructions/approvals/authorization-record-approval.json` contains `format` (`tifp-authorization-approval-v1`), approval ID, exact payload path/byte count/hash, approving privacy-authority identifier, decision (`approved` only), decision time, rationale, and `record_sha256`. The final `instructions/engagement-authorization.json` incorporates the payload fields and binds the payload plus approval-evidence artifact ID/path/byte count/hash, approval time, and its own `record_sha256`, using Section 6.1 rules. This evidence authorizes the engagement-authorization record only; it is distinct from publication-decision approval evidence.

- **TIFP-AUT-001 — SHALL**: Every run SHALL have one canonical engagement-authorization record before S1.
- **TIFP-AUT-002 — SHALL**: The authorization payload, approval evidence, and final authorization record SHALL be frozen and validated before S1; S2 SHALL hash-pin all three in the source manifest as `current_engagement_instruction` before S3 or any model invocation.
- **TIFP-AUT-003 — SHALL**: Approved audience and approved purpose checks SHALL use only the exact audience-role and purpose identifiers in the frozen authorization record.
- **TIFP-AUT-004 — SHALL**: A missing, stale, expired, ambiguous, or hash-mismatched authorization record SHALL fail closed.
- **TIFP-AUT-005 — SHALL**: Only the authorization record's named publication authority SHALL authorize an engagement publication decision.
- **TIFP-AUT-006 — SHALL**: Exposure SHALL be authorized only when audience, purpose, and their exact pair are all present in the final authorization record.
- **TIFP-AUT-007 — SHALL**: Authorization approval evidence SHALL conform to its canonical path, schema, serialization, and self-hash rules and bind the exact pre-approval payload path, byte count, and hash.
- **TIFP-AUT-008 — SHALL**: The final authorization record SHALL bind the exact payload and approval-evidence artifact IDs, paths, byte counts, and hashes.
- **TIFP-AUT-009 — SHALL**: A change to the authorization payload after approval SHALL invalidate the approval and final authorization record.
- **TIFP-AUT-010 — SHALL**: Only the protocol owner SHALL authorize release of a reusable protocol version.

## 4. Context isolation boundary

### 4.1 Exact allowlist

Every model-bound byte is authorized through exactly one of these five classes:

1. `protocol`: the hash-pinned client-neutral Protocol;
2. `source_manifest`: the frozen current-engagement Source Manifest;
3. `current_engagement_evidence`: manifest-listed current-engagement evidence;
4. `current_engagement_instruction`: manifest-listed current-engagement instructions; or
5. `bound_generated_artifact`: untrusted report, dossier, or validation-result data generated in this run, hash-pinned in the frozen stage-context manifest, allowed only at S7 Validate or S9 Review, and located under `deliverables/` or `review/validation-results/`.

A report or dossier can never be reclassified as an instruction. Class 5 is data, never authority or executable instructions, and is forbidden during extraction, analysis, and drafting.

- **TIFP-CTX-001 — SHALL**: Every model-bound byte SHALL resolve to exactly one allowlisted context class.
- **TIFP-CTX-002 — SHALL**: Every model-bound file SHALL match its approved SHA-256 hash.
- **TIFP-CTX-003 — SHALL**: Every model-bound evidence file SHALL be listed in the frozen source manifest.
- **TIFP-CTX-004 — SHALL**: Every model-bound instruction file SHALL be listed in the frozen source manifest.
- **TIFP-CTX-005 — SHALL**: Every model-bound evidence path SHALL resolve beneath the engagement evidence directory.
- **TIFP-CTX-006 — SHALL**: Every model-bound instruction path SHALL resolve beneath the engagement instructions directory.
- **TIFP-CTX-007 — SHALL**: Other-engagement information SHALL be absent from the evidence supplied to a model.
- **TIFP-CTX-008 — SHALL**: Other-engagement information SHALL be absent from every model context.
- **TIFP-CTX-009 — SHALL**: Prompt wording that asks a model to suppress supplied other-engagement information SHALL NOT satisfy TIFP-CTX-007 or TIFP-CTX-008.
- **TIFP-CTX-010 — SHALL**: A prior report SHALL NOT be supplied as model context.
- **TIFP-CTX-010A — SHALL**: A report or dossier SHALL NOT be classified as an instruction.
- **TIFP-CTX-010B — SHALL**: Bound-generated-artifact text SHALL be treated as untrusted data and SHALL NOT alter control instructions.
- **TIFP-CTX-010C — SHALL**: Bound-generated-artifact context SHALL be denied outside S7 and S9.
- **TIFP-CTX-011 — SHALL**: A sanitized structural specification derived from prior work SHALL be treated only as Protocol content.
- **TIFP-CTX-012 — SHALL**: A sanitized structural specification SHALL contain no prior-engagement identity.
- **TIFP-CTX-013 — SHALL**: A sanitized structural specification SHALL contain no prior-engagement fact.
- **TIFP-CTX-014 — SHALL**: A sanitized structural specification SHALL contain no prior-engagement quotation.
- **TIFP-CTX-015 — SHALL**: A sanitized structural specification SHALL contain no prior-engagement conclusion.

### 4.2 Denied ambient channels

- **TIFP-CTX-016 — SHALL**: Ambient conversation history SHALL be disabled.
- **TIFP-CTX-017 — SHALL**: Persistent model memory SHALL be disabled.
- **TIFP-CTX-018 — SHALL**: Ambient retrieval SHALL be disabled.
- **TIFP-CTX-019 — SHALL**: Prior-run output SHALL be denied.
- **TIFP-CTX-019A — SHALL**: At S7, a current bound generated artifact SHALL have been generated in this run and hash-bound in the current frozen S7 stage-context manifest.
- **TIFP-CTX-019C — SHALL**: At S9, a current bound generated artifact SHALL have been generated in this run and bound in this run's current subject set.
- **TIFP-CTX-019B — SHALL**: An earlier-run artifact SHALL NOT qualify as current.
- **TIFP-CTX-020 — SHALL**: Unmanifested template bodies SHALL be denied.
- **TIFP-CTX-021 — SHALL**: Unlisted files SHALL be denied.
- **TIFP-CTX-022 — SHALL**: Unlisted tool results SHALL be denied.
- **TIFP-CTX-023 — SHALL**: Unlisted retrieval results SHALL be denied.
- **TIFP-CTX-024 — SHALL**: Environment-variable content SHALL be denied from model context unless represented as an approved manifest-listed instruction.
- **TIFP-CTX-025 — SHALL**: Clipboard content SHALL be denied.
- **TIFP-CTX-026 — SHALL**: Automatic workspace discovery SHALL be disabled for model invocations.
- **TIFP-CTX-027 — SHALL**: Network retrieval SHALL be disabled for model invocations unless a later protocol version defines a manifest-before-use control.
- **TIFP-CTX-028 — SHALL**: The prohibited-identifier scan SHALL be treated as a secondary detection control.
- **TIFP-CTX-029 — SHALL**: The prohibited-identifier scan SHALL NOT substitute for positive path authorization.
- **TIFP-CTX-030 — SHALL**: The prohibited-identifier scan SHALL NOT substitute for class authorization.
- **TIFP-CTX-031 — SHALL**: The prohibited-identifier scan SHALL NOT substitute for hash authorization.

### 4.3 Engagement-wide provenance and ingestion

Other-engagement material is prohibited anywhere beneath the engagement root, whether or not selected for context and whether or not a prohibited name is detectable. This includes facts, metadata, attachments, relationships, links, embedded objects, and transformed fragments. `isolation-policy.json` uses canonical JSON and contains `format` (`tifp-isolation-policy-v1`), `engagement_id`, `allowed_root`, `allowed_provenance_authorities[]`, `allowed_audiences[]`, `prohibited_identifiers[]`, `prohibited_paths[]`, `upstream_system_checks[]`, `created_at`, `approved_by`, `state` (`frozen`), and `policy_sha256`. Each upstream check records system/custodian, source object ID, asserted engagement ID, checker, time, and result (`match` or `ambiguous`). Each source or instruction has a human `provenance_attestation` recording attestor, time, engagement ID, source of origin, upstream object ID, basis, and `current_engagement_only`.

- **TIFP-PROV-001 — SHALL**: Other-engagement material SHALL be absent everywhere beneath the engagement root.
- **TIFP-PROV-002 — SHALL**: Other-engagement material SHALL NOT be listed in the source manifest.
- **TIFP-PROV-003 — SHALL**: Each source and instruction SHALL have a human provenance attestation.
- **TIFP-PROV-004 — SHALL**: Each source and instruction SHALL pass an upstream source-of-origin engagement check before ingestion.
- **TIFP-PROV-005 — SHALL**: Ambiguous provenance SHALL fail closed before ingestion.
- **TIFP-PROV-006 — SHALL**: The isolation policy SHALL conform to its declared schema and be frozen before ingestion.
- **TIFP-PROV-007 — SHALL**: Ingestion SHALL inspect filenames, paths, metadata, attachments, relationships, links, and embedded objects before copying.
- **TIFP-PROV-008 — SHALL**: A clean semantic or identifier scan SHALL NOT establish provenance.
- **TIFP-PROV-009 — SHALL**: Structural guarantees SHALL be limited to authorized provenance, path, class, hash, and outbound bytes.
- **TIFP-PROV-010 — SHALL**: Discovered human misattestation SHALL be recorded as a human governance failure and blocker.

## 5. Source manifest schema and freezing

`source-manifest.json` is a canonical serialized object with this logical schema.

Its canonical serialization and self-hash use the JSON rules in Section 6.1.

| Field | Cardinality | Meaning |
|---|---:|---|
| `format` | 1 | `tifp-source-manifest-v1` |
| `engagement_id` | 1 | Non-identifying stable engagement key |
| `allowed_root` | 1 | Canonical absolute engagement root |
| `created_at` | 1 | Snapshot time |
| `manifest_state` | 1 | `frozen` |
| `protocol_id` | 1 | `TIFP` |
| `protocol_version` | 1 | Semantic version |
| `protocol_sha256` | 1 | Approved protocol hash |
| `isolation_policy_sha256` | 1 | Isolation-policy hash |
| `files[]` | 1..n | File records |
| `manifest_sha256` | 1 | Hash of canonical form with this field omitted |

Each `files[]` record has: `file_id`, `relative_path`, `bytes`, `sha256`, `source_class` (`primary`, `derivative`, or `instruction`), `media_type`, `retrieved_at`, `source_date` if known, `duration_ms` or `page_range` as applicable, `diarization_reliability` (`reliable`, `unreliable`, `not_applicable`, or `unknown`), `parent_file_ids[]`, `provenance_attestation`, `upstream_check_id`, and `transformations[]`. Each transformation has `operation`, `tool`, `performed_at`, `input_sha256`, and `output_sha256`.

- **TIFP-MAN-001 — SHALL**: The source manifest SHALL conform to the declared schema.
- **TIFP-MAN-002 — SHALL**: Each manifest file record SHALL use a stable unique `file_id`.
- **TIFP-MAN-003 — SHALL**: Each manifest file record SHALL include a root-relative path.
- **TIFP-MAN-004 — SHALL**: Each manifest file record SHALL include an exact byte count.
- **TIFP-MAN-005 — SHALL**: Each manifest file record SHALL include a SHA-256 hash.
- **TIFP-MAN-006 — SHALL**: Each manifest file record SHALL declare one source class.
- **TIFP-MAN-007 — SHALL**: Each primary transcript record SHALL include its duration or page range.
- **TIFP-MAN-008 — SHALL**: Each source record SHALL include its retrieval timestamp.
- **TIFP-MAN-009 — SHALL**: Each transcript record SHALL declare diarization reliability.
- **TIFP-MAN-010 — SHALL**: Each transformed source SHALL record transformation lineage.
- **TIFP-MAN-011 — SHALL**: Each derivative source SHALL identify its parent source records.
- **TIFP-MAN-012 — SHALL**: The complete primary transcript SHALL be preserved.
- **TIFP-MAN-013 — SHALL**: Canonical source metadata SHALL be preserved.
- **TIFP-MAN-014 — SHALL**: The source manifest SHALL be frozen before analysis begins.
- **TIFP-MAN-015 — SHALL**: A post-freeze source addition SHALL trigger a new run.
- **TIFP-MAN-016 — SHALL**: A separately attributed late input SHALL be disclosed as outside the original snapshot.
- **TIFP-MAN-017 — SHALL**: A late input used in analysis SHALL be included only in a newly frozen manifest.
- **TIFP-MAN-018 — SHALL**: Each file record SHALL include a complete provenance attestation and passing upstream-check reference.

## 6. Stage-context and outbound-context manifests

### 6.1 Canonical stage-context schema

All manifests and canonical control records in this protocol use UTF-8 JSON, NFC strings, lexicographically sorted object keys, preserved array order, no insignificant whitespace, and one final LF. Integers are base-10 without leading zero; floats and duplicate keys are forbidden. Each schema declares exactly one top-level self-hash field, such as `manifest_sha256`, `plan_sha256`, `event_sha256`, or `record_sha256`; its value is lowercase SHA-256 of the canonical serialization with that declared self-hash field omitted. No other field is omitted.

A stage-context manifest is stored at `work/stage-context-manifests/stage-context-<stage>-<manifest_id>.json`; reviewer controls use `review/reviewer-controls/reviewer-control-<manifest_id>.json`. Required fields are `format` (`tifp-stage-context-manifest-v1`), `manifest_id`, `run_id`, `engagement_id`, `stage`, `created_at`, `state` (`frozen`), `protocol` (ID/version/canonical path/bytes/hash), `source_manifest` (ID/canonical path/bytes/hash), `subject_set_manifest` (ID/path/bytes/hash required at S9, otherwise null), `reviewer_control_manifest_sha256` (required at S9, otherwise null), `entries[]`, and `manifest_sha256`. Each ordered entry has artifact ID, one of the five exact context-class tokens, canonical and relative path, bytes, hash, source-manifest file ID when applicable, `generated_in_run_id` for class 5, allowed stage, and `data_not_instructions` true for class 5.

A protocol-development context manifest is stored beneath a newly created client-neutral `<protocol-development-root>/context-manifests/` and uses `format` (`tifp-protocol-development-context-v1`), `lifecycle_scope` (`protocol_development`), manifest/invocation IDs, `stage` (`external_protocol_challenge`), creation time, frozen state, protocol ID/version/path/bytes/hash, neutral-governing-requirements manifest path/bytes/hash, ordered `entries[]`, and `manifest_sha256`. Each entry uses exactly `protocol_specification` or `client_neutral_governing_requirement` and records artifact ID, canonical path, byte count, and hash. It has no engagement ID, source manifest, engagement instruction, generated artifact, or dossier field.

- **TIFP-SCM-001 — SHALL**: Every engagement-run model stage SHALL have one frozen engagement stage-context manifest before payload serialization.
- **TIFP-SCM-001A — SHALL**: Every protocol-development challenge SHALL have one frozen protocol-development context manifest before payload serialization.
- **TIFP-SCM-002 — SHALL**: Each stage-context manifest SHALL conform to the declared schema, serialization, hash, and path rules.
- **TIFP-SCM-003 — SHALL**: Each engagement stage-context manifest SHALL bind the immutable source-manifest path and hash without modifying or superseding it.
- **TIFP-SCM-004 — SHALL**: Every entry SHALL pass path, class, hash, byte-count, provenance, and stage authorization.
- **TIFP-SCM-005 — SHALL**: Each class-5 entry SHALL identify this run and set `data_not_instructions` true.
- **TIFP-SCM-006 — SHALL**: Reviewer control instructions SHALL be frozen before subject-set binding.

### 6.2 Outbound context manifest

Before each model invocation, the context builder emits `work/context-manifests/context-manifest-<invocation_id>.json`.

Required top-level fields are: `format` (`tifp-outbound-context-manifest-v1`), `lifecycle_scope`, `invocation_id`, `stage`, `created_at`, `stage_context_manifest_path`, `stage_context_manifest_sha256`, `source_manifest_sha256` for engagement scope or null for protocol-development scope, `protocol_sha256`, `segments[]`, `serialized_payload_bytes`, `serialized_payload_sha256`, and `manifest_sha256`. Each ordered segment contains `ordinal`, `context_class`, `artifact_id`, `source_sha256`, `source_byte_start`, `source_byte_end_exclusive`, `payload_byte_start`, `payload_byte_end_exclusive`, and `segment_sha256`. The manifest itself is stored outside the outbound payload unless it was already authorized as a current-engagement instruction.

- **TIFP-OCM-001 — SHALL**: The context builder SHALL emit an outbound context manifest before every model invocation.
- **TIFP-OCM-002 — SHALL**: The outbound context manifest SHALL identify every outbound segment in payload order.
- **TIFP-OCM-003 — SHALL**: Each engagement-run outbound segment SHALL identify one of the five Section 4 allowlisted context classes.
- **TIFP-OCM-003A — SHALL**: Each protocol-development outbound segment SHALL identify exactly one protocol-development class: `protocol_specification` or `client_neutral_governing_requirement`.
- **TIFP-OCM-004 — SHALL**: Each outbound segment SHALL identify an exact source byte range.
- **TIFP-OCM-005 — SHALL**: Each outbound segment SHALL identify an exact payload byte range.
- **TIFP-OCM-006 — SHALL**: Each outbound segment SHALL include its SHA-256 hash.
- **TIFP-OCM-007 — SHALL**: The outbound context manifest SHALL include the exact serialized payload byte count.
- **TIFP-OCM-008 — SHALL**: The outbound context manifest SHALL include the exact serialized payload SHA-256 hash.
- **TIFP-OCM-009 — SHALL**: Context authorization SHALL be completed against the serialized outbound payload.
- **TIFP-OCM-010 — SHALL**: A payload containing any byte not covered by one authorized segment SHALL fail before invocation.
- **TIFP-OCM-011 — SHALL**: Overlapping payload segment ranges SHALL fail before invocation.
- **TIFP-OCM-012 — SHALL**: A gap between payload segment ranges SHALL fail before invocation.
- **TIFP-OCM-013 — SHALL**: The outbound manifest SHALL reference its frozen authorizing stage-context manifest by canonical path and hash.
- **TIFP-OCM-014 — SHALL**: Every engagement-run outbound segment SHALL resolve to exactly one engagement stage-context entry.
- **TIFP-OCM-014A — SHALL**: Every protocol-development outbound segment SHALL resolve to exactly one protocol-development-context entry of the same class, path, byte count, and hash.
- **TIFP-OCM-015 — SHALL**: The outbound context manifest SHALL conform to its canonical path, schema, serialization, and self-hash rules.

### 6.3 Model-response capture

Every model invocation produces canonical `work/context-manifests/response-<invocation_id>.json` with `format` (`tifp-model-response-v1`), invocation/run/stage IDs, request outbound-context-manifest hash, started/completed times, status (`complete` or `error`), provider/model disclosure, raw-response byte count and SHA-256 when bytes are returned, error envelope when status is `error`, parser/tool/version, transformations, generated artifact IDs/paths/byte counts/hashes, and `record_sha256`. Raw response bytes are quarantined as current-engagement generated data and never become evidence or model context. Only parsed generated artifacts may receive class-5 authorization at S7 or S9.

When bytes are returned, they are preserved immutably at `work/responses/raw/<invocation_id>.bin`. Canonical `work/responses/raw-response-manifest.json` with `format` (`tifp-raw-response-manifest-v1`) lists invocation ID, raw path, byte count/hash, response-capture path/hash, access-policy role IDs, retention/disposal trigger, and manifest self-hash. Every read is appended to `work/responses/raw-response-access.jsonl` with actor role, purpose, time, invocation ID, and prior-entry hash. The dossier references the manifest path/hash without embedding raw bytes.

- **TIFP-RSP-001 — SHALL**: Every model invocation SHALL produce one canonical response-capture record.
- **TIFP-RSP-002 — SHALL**: The response record SHALL bind the exact outbound-context-manifest hash.
- **TIFP-RSP-003 — SHALL**: A completed response SHALL record raw returned byte count and SHA-256.
- **TIFP-RSP-004 — SHALL**: An errored or incomplete response SHALL record an error envelope and SHALL NOT be treated as complete.
- **TIFP-RSP-005 — SHALL**: Every transformation from returned bytes to a generated artifact SHALL record tool, version, input hash, output hash, and time.
- **TIFP-RSP-006 — SHALL**: Generated artifact hashes SHALL match the response record before stage use.
- **TIFP-RSP-007 — SHALL**: Raw response bytes SHALL NOT enter evidence or model context.
- **TIFP-RSP-007A — SHALL**: Only parsed generated artifacts whose path, bytes, and hash are recorded in the response-capture record SHALL be eligible to enter S7 or S9 as stage-authorized class-5 data.

### 6.4 Model-service boundary

Before transmission, canonical JSON at `work/context-manifests/model-service-boundary-<invocation_id>.json`, using `format` (`tifp-model-service-boundary-v1`) and Section 6.1 self-hash rules, identifies the invocation/run/stage IDs, context-builder process boundary, disclosed provider/model service boundary, provider/model IDs, destination endpoint or service identifier, transport protection, payload classes and byte count, outbound-context-manifest hash, provider retention/training policy status, authorized provider-side access roles, data-region commitment when applicable, contract/control-evidence hashes, approval owner, `created_at`, and `record_sha256`.

- **TIFP-RSP-008 — SHALL**: The model-service-boundary record SHALL freeze before each invocation and bind the exact outbound-context-manifest hash.
- **TIFP-RSP-009 — SHALL**: Payload transmission SHALL use authenticated encrypted transport to the exact disclosed destination.
- **TIFP-RSP-010 — SHALL**: Model-service controls SHALL count as approved only when the exact provider/model/destination matches one `approved_model_services[]` entry, training use is false, retention does not exceed its maximum, provider-side roles and region are subsets of its allowlists, transport meets its minimum, and the final authorization record carries the privacy approver's valid approval; any unknown or mismatch SHALL block transmission.
- **TIFP-RSP-011 — SHALL**: The response-capture record SHALL bind the exact model-service-boundary-record hash.
- **TIFP-RSP-012 — SHALL**: Provider credentials or secret values SHALL NOT enter the boundary record, evidence, model context, report, dossier, or audit package.
- **TIFP-RSP-013 — SHALL**: Returned raw-response bytes SHALL be written once to the canonical raw-response path and their path, byte count, and hash SHALL match the response-capture record.
- **TIFP-RSP-014 — SHALL**: The raw-response manifest SHALL conform to its canonical path, schema, serialization, and self-hash rules and enumerate every returned raw response exactly once.
- **TIFP-RSP-015 — SHALL**: Raw-response access SHALL be restricted to roles named in the manifest and every read SHALL append a predecessor-bound access event.
- **TIFP-RSP-016 — SHALL**: The subject-set manifest and dossier SHALL bind the exact raw-response-manifest path, byte count, and hash without embedding raw bytes.
- **TIFP-RSP-017 — SHALL**: Raw-response bytes SHALL remain available under restricted access until S11 executes or schedules their authorized retention, quarantine, or disposal action.
- **TIFP-RSP-018 — SHALL**: The closure record SHALL bind the final raw-response-manifest hash and the executed or scheduled raw-response retention action.

## 7. Stable IDs and referential integrity

Canonical forms are `CLM-000001` for claims, `EVD-000001` for evidence, `QTE-000001` for quotes, `FNT-000001` for footnotes/endnotes, `SRC-000001` for sources, `REV-000001` for review records, and `ADV-000001` for advisories. IDs are assigned once within an engagement.

- **TIFP-ID-001 — SHALL**: Every material report claim SHALL have a stable claim ID.
- **TIFP-ID-002 — SHALL**: Every evidence-ledger entry SHALL have a stable evidence ID.
- **TIFP-ID-003 — SHALL**: Every direct quotation SHALL have a stable quote ID.
- **TIFP-ID-004 — SHALL**: Every report note SHALL have a stable footnote ID.
- **TIFP-ID-005 — SHALL**: An assigned stable ID SHALL NOT be reused.
- **TIFP-ID-006 — SHALL**: An assigned stable ID SHALL NOT be renumbered.
- **TIFP-ID-007 — SHALL**: Deletion of an identified record SHALL leave a tombstone in the dossier.
- **TIFP-ID-008 — SHALL**: Every report footnote ID SHALL resolve to exactly one dossier footnote record.
- **TIFP-ID-009 — SHALL**: Every dossier evidence reference SHALL resolve to exactly one evidence-ledger entry.
- **TIFP-ID-010 — SHALL**: Every source reference SHALL resolve to exactly one frozen manifest record.
- **TIFP-ID-011 — SHALL**: Referential-integrity failure SHALL be a publication blocker.

## 8. Ordered stages and gates

The productive stages execute in this order. A failed gate prevents entry to the next productive stage and instead takes the failure-control transition directly to S11 Close with a denied terminal state.

1. **S0 Initialize** — create workspace and isolation policy. **G0:** workspace valid.
2. **S1 Ingest** — copy complete current-engagement sources and instructions. **G1:** paths and classes valid.
3. **S2 Freeze** — hash sources and freeze manifest. **G2:** manifest valid and immutable.
4. **S3 Build context** — create exact allowlisted payload and outbound context manifest. **G3:** isolation passes.
5. **S4 Extract** — populate evidence and quote ledgers before synthesis. **G4:** source locations resolve.
6. **S5 Analyze** — classify claims, alternatives, contradictions, assumptions, and unknowns. **G5:** analysis completeness passes.
7. **S6 Draft** — create report and dossier candidates separately; freeze all dossier-resident validation inputs and findings that can exist before exact-candidate validation. **G6:** report/dossier schema passes.
8. **S7 Validate** — validate the exact immutable report and dossier candidates using frozen validation context; write the G7 result only to a separate immutable validation artifact outside the dossier. **G7:** checks pass and the external G7 artifact freezes.
9. **S8 Bind** — freeze the subject-set manifest, including pre-frozen reviewer controls, the external G7 artifact, and the requirements verification plan. **G8:** all subject artifacts become immutable.
10. **S9 Review** — review the subject set and append a disposition outside the dossier. **G9:** disposition passes and binds the subject-set hash.
11. **S10 Decide/Publish** — append a publication record outside the dossier, then copy exact approved bytes. **G10:** exact subject set authorized.
12. **S11 Close** — verify hashes and execute or schedule owner-defined retention/disposal. **G11:** terminal state is `closed_published`, `closed_denied`, or `closed_disposed`.

- **TIFP-STG-001 — SHALL**: Stages SHALL execute in the stated order.
- **TIFP-STG-002 — SHALL**: G0 through G6 results SHALL be frozen in the dossier or pre-G7 bound validation artifacts before S7.
- **TIFP-STG-002C — SHALL**: The final G7 result SHALL exist only in a separate immutable validation artifact outside the dossier.
- **TIFP-STG-002D — SHALL**: S8 SHALL bind the exact report, exact dossier, and external G7 validation artifact without editing the report or dossier.
- **TIFP-STG-002A — SHALL**: G8 through G11 results SHALL be append-only control records outside the dossier.
- **TIFP-STG-002B — SHALL**: Appending a control record SHALL NOT modify any reviewed artifact.
- **TIFP-STG-003 — SHALL**: A failed gate SHALL prevent entry to the next productive stage.
- **TIFP-STG-003A — SHALL**: A failed, errored, or incomplete mandatory gate from G0 through G10 SHALL transition directly to S11 Close-Denied.
- **TIFP-STG-003B — SHALL**: The S11 failure-control transition SHALL execute or schedule retention, quarantine, access-revocation, and disposal actions without permitting publication.
- **TIFP-STG-003C — SHALL**: A failed, errored, or incomplete G11 SHALL place the run in nonterminal `closure_exception` state and SHALL NOT transition from S11 to S11.
- **TIFP-STG-003D — SHALL**: `closure_exception` SHALL append an immutable exception record, quarantine affected publication access, identify an owner and remediation, and retry G11.
- **TIFP-STG-003E — SHALL**: A run in `closure_exception` SHALL NOT declare a terminal state or conformance until a retry of G11 passes.
- **TIFP-STG-004 — SHALL**: Evidence extraction SHALL precede synthesis.
- **TIFP-STG-005 — SHALL**: Analysis SHALL be distinct from drafting.
- **TIFP-STG-006 — SHALL**: Drafting SHALL be distinct from adversarial review.
- **TIFP-STG-007 — SHALL**: Validation SHALL inspect the exact deliverable candidates.
- **TIFP-STG-008 — SHALL**: Review SHALL inspect the exact bound artifact set.
- **TIFP-STG-009 — SHALL**: The lifecycle SHALL terminate in exactly one declared terminal state.
- **TIFP-STG-010 — SHALL**: Remediation changing a bound artifact SHALL create a new subject set followed by fresh validation and review.

## 9. Evidence ledger schema

Each evidence-ledger record contains: `evidence_id`, `source_id`, exact `location` (timestamp label/span or page/line span), `verbatim_excerpt`, `speaker_attribution`, `speech_act`, `classification`, `summary`, `context_before`, `context_after`, `claim_ids[]`, `counterevidence_ids[]`, `dependence_group_id`, `analyst`, and `extracted_at`. `classification` is one of `stated_fact`, `interpretation`, `inference`, `unknown`, `disputed`, `counterevidence`, or `proposed_action`. `speech_act` is one of `adopted_decision`, `hypothetical`, `question`, `proposed_wording`, `rejected_option`, `reported_view`, or `other`.

- **TIFP-LED-001 — SHALL**: Every evidence-ledger record SHALL conform to the evidence-ledger schema.
- **TIFP-LED-002 — SHALL**: Every evidence-ledger record SHALL identify an exact source location.
- **TIFP-LED-003 — SHALL**: Every evidence-ledger record SHALL carry exactly one claim classification.
- **TIFP-LED-004 — SHALL**: Every evidence-ledger record SHALL carry exactly one speech-act classification.
- **TIFP-LED-005 — SHALL**: Materially relevant surrounding context SHALL be preserved in the ledger.
- **TIFP-LED-005A — SHALL**: Initial extraction SHALL capture the complete target speaker turn plus immediately preceding and following turns when available.
- **TIFP-LED-005B — SHALL**: The analyst SHALL expand through every meaning-changing referent, negation, qualification, condition, uncertainty marker, speech act, and linked contradiction.
- **TIFP-LED-005C — SHALL**: The analyst SHALL record context boundaries and expansion rationale.
- **TIFP-LED-005D — SHALL**: An independent reviewer SHALL verify context sufficiency against the primary source.
- **TIFP-LED-006 — SHALL**: Contradictory evidence SHALL be linked to the affected claim.
- **TIFP-LED-007 — SHALL**: Counterevidence SHALL be represented in the ledger.
- **TIFP-LED-008 — SHALL**: Competing interpretations SHALL be represented as distinct records.
- **TIFP-LED-009 — SHALL**: An unknown SHALL NOT be encoded as a stated fact.
- **TIFP-LED-010 — SHALL**: A proposed action SHALL NOT be encoded as an adopted decision.

## 10. Analysis requirements

- **TIFP-ANL-001 — SHALL**: Analysis SHALL distinguish stated facts from interpretations.
- **TIFP-ANL-002 — SHALL**: Analysis SHALL distinguish interpretations from inferences.
- **TIFP-ANL-003 — SHALL**: Analysis SHALL identify unknowns.
- **TIFP-ANL-004 — SHALL**: Analysis SHALL identify disputed claims.
- **TIFP-ANL-005 — SHALL**: Analysis SHALL identify counterevidence.
- **TIFP-ANL-006 — SHALL**: Analysis SHALL distinguish adopted decisions from hypotheticals.
- **TIFP-ANL-007 — SHALL**: Analysis SHALL distinguish adopted decisions from questions.
- **TIFP-ANL-008 — SHALL**: Analysis SHALL distinguish adopted decisions from proposed wording.
- **TIFP-ANL-009 — SHALL**: Analysis SHALL distinguish adopted decisions from rejected options.
- **TIFP-ANL-010 — SHALL**: Analysis SHALL distinguish participant statements from interpretations attributed to absent stakeholders.
- **TIFP-ANL-011 — SHALL**: Materially different strategic end states SHALL remain distinct.
- **TIFP-ANL-012 — SHALL**: Competing strategic end states SHALL NOT be blended into asserted consensus.
- **TIFP-ANL-013 — SHALL**: Conclusions SHALL be bounded to represented participants and evidence.
- **TIFP-ANL-014 — SHALL**: A meeting-level observation SHALL NOT be generalized organization-wide without independent evidence.
- **TIFP-ANL-015 — SHALL**: Each recommendation SHALL disclose its assumptions.
- **TIFP-ANL-016 — SHALL**: Each recommendation SHALL identify evidence that would falsify it.
- **TIFP-ANL-017 — SHALL**: A decision-relevant forecast SHALL state its evidence basis.
- **TIFP-ANL-018 — SHALL**: A decision-relevant forecast SHALL include outside-view treatment.
- **TIFP-ANL-019 — SHALL**: An unsupported forecast SHALL be omitted or labeled as a planning assumption.
- **TIFP-ANL-020 — SHALL**: An unsupported ROI estimate SHALL be omitted or labeled as a planning assumption.
- **TIFP-ANL-021 — SHALL**: An unsupported budget SHALL be omitted or labeled as a planning assumption.
- **TIFP-ANL-022 — SHALL**: An unsupported timeline SHALL be omitted or labeled as a planning assumption.
- **TIFP-ANL-023 — SHALL**: An unsupported adoption estimate SHALL be omitted or labeled as a planning assumption.
- **TIFP-ANL-024 — SHALL**: An unsupported probability SHALL be omitted or labeled as a planning assumption.
- **TIFP-ANL-025 — SHALL**: Every proposed pilot SHALL state what it can establish.
- **TIFP-ANL-026 — SHALL**: Every proposed pilot SHALL state what it cannot establish.

## 11. Source-dependence rule

- **TIFP-DEP-001 — SHALL**: A generated summary SHALL be classified as a derivative source.
- **TIFP-DEP-002 — SHALL**: A generated highlight SHALL be classified as a derivative source.
- **TIFP-DEP-003 — SHALL**: A meeting note derived from a transcript SHALL be classified as a derivative source.
- **TIFP-DEP-004 — SHALL**: Verification against a parent primary source SHALL NOT make a derivative source independent.
- **TIFP-DEP-005 — SHALL**: Sources sharing a common primary source SHALL share a dependence-group identifier.
- **TIFP-DEP-006 — SHALL**: Sources in one dependence group SHALL count as one corroborating lineage.
- **TIFP-DEP-007 — SHALL**: A corroboration claim SHALL require evidence independent of the common primary source.
- **TIFP-DEP-008 — SHALL**: Derivative accuracy SHALL be recorded separately from source independence.

## 12. Quote normalization and validation

The validator compares a report quotation with preserved primary-source spans. The normalization function applies only the following operations, in order: Unicode NFC; quotation-mark mapping; line-break-to-space conversion; repeated-whitespace collapse. The complete quotation-mark mapping is `U+2018`, `U+2019`, `U+201A`, `U+201B`, `U+2039`, and `U+203A` to ASCII `'`, and `U+201C`, `U+201D`, `U+201E`, `U+201F`, `U+00AB`, and `U+00BB` to ASCII `"`. No other punctuation is normalized. The function does not remove leading or trailing whitespace except as an effect of comparison boundaries selected in the source span.

- **TIFP-QTE-001 — SHALL**: Every direct quotation SHALL map to a primary-source span.
- **TIFP-QTE-002 — SHALL**: Quote validation SHALL apply Unicode NFC normalization.
- **TIFP-QTE-003 — SHALL**: Quote validation SHALL normalize typographic quotation marks.
- **TIFP-QTE-004 — SHALL**: Quote validation SHALL convert line breaks to spaces.
- **TIFP-QTE-005 — SHALL**: Quote validation SHALL collapse repeated whitespace.
- **TIFP-QTE-006 — SHALL**: Quote normalization SHALL NOT alter lexical words.
- **TIFP-QTE-007 — SHALL**: Quote normalization SHALL NOT alter numbers.
- **TIFP-QTE-008 — SHALL**: Quote normalization SHALL NOT alter negation.
- **TIFP-QTE-009 — SHALL**: Quote normalization SHALL NOT reorder words.
- **TIFP-QTE-010 — SHALL**: Quote normalization SHALL NOT remove disfluencies.
- **TIFP-QTE-011 — SHALL**: Quote normalization SHALL NOT change meaning.
- **TIFP-QTE-012 — SHALL**: An omission SHALL use an explicit ellipsis.
- **TIFP-QTE-013 — SHALL**: Each ellipsis SHALL link to the preserved omitted source span.
- **TIFP-QTE-014 — SHALL**: A clarification SHALL use brackets.
- **TIFP-QTE-015 — SHALL**: Each bracketed clarification SHALL be separately reviewable.
- **TIFP-QTE-016 — SHALL**: A paraphrase SHALL NOT appear inside quotation marks.
- **TIFP-QTE-017 — SHALL**: A quotation SHALL preserve its speech-act context.
- **TIFP-QTE-018 — SHALL**: A quotation SHALL preserve exact existing timestamp labels.
- **TIFP-QTE-019 — SHALL**: A compressed timestamp range spanning nonexistent labels SHALL NOT be created.
- **TIFP-QTE-019A — SHALL**: Reliable diarization SHALL require authenticated identity or designated-human recording verification for the cited span with no unresolved conflict.
- **TIFP-QTE-019B — SHALL**: Automated-only, ambiguous, unknown, or conflicting assignment SHALL be unreliable.
- **TIFP-QTE-020 — SHALL**: A quotation from unreliable diarization SHALL be attributed to the meeting or transcript.
- **TIFP-QTE-021 — SHALL**: A quotation from unreliable diarization SHALL NOT be attributed to a named speaker.
- **TIFP-QTE-022 — SHALL**: Any undocumented quote transformation SHALL fail validation.

The quote-ledger record contains `quote_id`, `claim_id`, `source_id`, `source_spans[]`, `source_text`, `rendered_text`, `normalization_operations[]`, `ellipsis_spans[]`, `bracket_clarifications[]`, `speech_act`, `attribution`, `validator_result`, and `reviewer_result`.

- **TIFP-QTE-023 — SHALL**: Every direct quotation SHALL have a complete quote-ledger record.
- **TIFP-QTE-024 — SHALL**: Every direct quotation SHALL pass automated validation.
- **TIFP-QTE-025 — SHALL**: Every direct quotation SHALL pass validation by the independent-review role defined in Section 2 under the isolation controls in Section 15.

## 13. Client-facing Initial Findings Report rules

The default structure is: title and purpose; executive findings; observations; implications; recommendations; unresolved questions; and concise documentation (scope, sources, methods, limitations, and note list).

- **TIFP-RPT-001 — SHALL**: The report SHALL communicate findings in prose that passes TIFP-RPT-001A.
- **TIFP-RPT-001A — SHALL**: Report prose SHALL use complete sentences except in headings, tables, and intentional lists; define specialized terms at first use; avoid internal protocol IDs, grading labels, and dossier jargon; identify observation, recommendation, assumption, limitation, and unknown status; and contain no unresolved ambiguity that could change a client decision-maker's interpretation.
- **TIFP-RPT-002 — SHALL**: The report SHALL state its evidence scope.
- **TIFP-RPT-003 — SHALL**: The report SHALL separate observations from recommendations through prose and section structure.
- **TIFP-RPT-004 — SHALL**: Material report claims SHALL use footnotes or endnotes.
- **TIFP-RPT-004A — SHALL**: Every rendered client note SHALL include a source label and an existing timestamp, page, or line span intelligible without the dossier.
- **TIFP-RPT-004B — SHALL**: An opaque internal ID alone SHALL NOT satisfy the client-note locator requirement.
- **TIFP-RPT-005 — SHALL**: Each material report claim SHALL map to at least one exact dossier entry unless identified as a recommendation or assumption.
- **TIFP-RPT-006 — SHALL**: A recommendation SHALL be visibly expressed as a recommendation.
- **TIFP-RPT-007 — SHALL**: A planning assumption SHALL be visibly expressed as an assumption.
- **TIFP-RPT-008 — SHALL**: A quotation SHALL appear only when exact wording materially improves meaning or clarity over faithful paraphrase.
- **TIFP-RPT-008A — SHALL**: The drafter SHALL record a rationale comparing the quotation with a faithful paraphrase.
- **TIFP-RPT-008B — SHALL**: The independent reviewer SHALL accept that rationale before publication.
- **TIFP-RPT-008C — SHALL**: Without reviewer acceptance, the report SHALL use a paraphrase with a client-intelligible locator note.
- **TIFP-RPT-008D — SHALL**: “Materially improves” SHALL pass only when the rationale and reviewer identify at least one of these conditions: the exact wording is itself the subject of the finding; paraphrase would remove or alter a decision-relevant qualifier, uncertainty, condition, chronology, commitment, technical term, or speech act; or the exact wording is necessary to verify a contested interpretation.
- **TIFP-RPT-008E — SHALL**: The quote rationale SHALL include the proposed faithful paraphrase and identify the exact semantic information that paraphrase would lose or alter.
- **TIFP-RPT-008F — SHALL**: If no TIFP-RPT-008D condition is demonstrated, the quotation-selection requirement SHALL fail and the report SHALL use paraphrase.
- **TIFP-RPT-009 — SHALL**: A direct quotation SHALL NOT be used merely to demonstrate that evidence exists.
- **TIFP-RPT-010 — SHALL**: Evidence grades SHALL NOT appear inline in the argument.
- **TIFP-RPT-011 — SHALL**: Provenance labels SHALL NOT appear inline in the argument.
- **TIFP-RPT-012 — SHALL**: Verdict labels SHALL NOT appear inline in the argument.
- **TIFP-RPT-013 — SHALL**: Corpus identifiers SHALL NOT appear in client-facing prose.
- **TIFP-RPT-014 — SHALL**: Citation inventories SHALL NOT interrupt the argument.
- **TIFP-RPT-015 — SHALL**: Raw file paths SHALL NOT appear in the report.
- **TIFP-RPT-016 — SHALL**: Model names SHALL NOT appear in the report.
- **TIFP-RPT-017 — SHALL**: Provider names SHALL NOT appear in the report.
- **TIFP-RPT-018 — SHALL**: Agent names SHALL NOT appear in the report.
- **TIFP-RPT-019 — SHALL**: Verifier mechanics SHALL NOT appear in the report.
- **TIFP-RPT-020 — SHALL**: Internal drafting instructions SHALL NOT appear in the report.
- **TIFP-RPT-021 — SHALL**: Prior-engagement template content SHALL NOT appear in the report.
- **TIFP-RPT-022 — SHALL**: Methods SHALL appear in one dedicated section of no more than 500 words, after findings and recommendations and before notes or appendices.
- **TIFP-RPT-023 — SHALL**: Limitations SHALL appear in one dedicated section of no more than 500 words, after methods and before notes or appendices.
- **TIFP-RPT-024 — SHALL**: Unresolved questions SHALL appear in a dedicated section or appendix.
- **TIFP-RPT-025 — SHALL**: The central findings SHALL be understandable without access to the dossier.

## 14. Evidence and Verification Dossier schema

The dossier contains these sections: artifact metadata; frozen source manifest; isolation-policy summary; outbound context manifests; evidence ledger; quote ledger; claim registry; claim-to-source map; source-dependence map; contradiction and counterevidence register; assumptions; unresolved questions; citation inventory; validation inputs and findings available before final exact-candidate validation; pre-binding advisory dispositions; subject-set preparation metadata; and change log. The final G7 validation artifact, review dispositions, publication records, closure record, and requirements trace are separate immutable audit-package records outside the dossier.

Each claim-registry record contains `claim_id`, `report_location`, `claim_text`, `claim_kind`, `evidence_ids[]`, `footnote_ids[]`, `status` (`supported`, `qualified`, `disputed`, or `unsupported`), `assumptions[]`, `falsifiers[]`, and `materiality`. Each footnote record contains `footnote_id`, `claim_ids[]`, `evidence_ids[]`, `rendered_note`, and the hashes of referenced ledger records.

- **TIFP-DOS-001 — SHALL**: The dossier SHALL contain every listed dossier section.
- **TIFP-DOS-002 — SHALL**: Each material report claim SHALL have a claim-registry record.
- **TIFP-DOS-003 — SHALL**: Each claim-registry record SHALL declare one evidence status.
- **TIFP-DOS-004 — SHALL**: Every supported claim SHALL link to exact source locations.
- **TIFP-DOS-005 — SHALL**: Every qualified claim SHALL record its qualification.
- **TIFP-DOS-006 — SHALL**: Every disputed claim SHALL record the dispute.
- **TIFP-DOS-007 — SHALL**: Every unsupported claim SHALL be removed from report assertions or visibly represented as an assumption, recommendation, or unresolved question.
- **TIFP-DOS-008 — SHALL**: The dossier SHALL preserve contradictions.
- **TIFP-DOS-009 — SHALL**: The dossier SHALL preserve counterevidence.
- **TIFP-DOS-010 — SHALL**: The dossier SHALL preserve unresolved questions.
- **TIFP-DOS-011 — SHALL**: The dossier SHALL preserve assumptions.
- **TIFP-DOS-012 — SHALL**: The dossier SHALL preserve only validation inputs and findings finalized before final G7 exact-candidate validation.
- **TIFP-DOS-012A — SHALL**: The final G7 result SHALL NOT be added to or embedded in the validated dossier.
- **TIFP-DOS-013 — SHALL**: Review findings and dispositions SHALL be preserved only in immutable external review records.
- **TIFP-DOS-014 — SHALL**: Dossier access SHALL be governed independently from report access.
- **TIFP-DOS-015 — SHALL**: The report and dossier SHALL remain separate artifacts.

## 15. Reviewer independence and review schema

Before S8, reviewer controls and the requirements verification plan are frozen. S8 then freezes canonical `review/subject-sets/subject-set-<id>.json` using Section 6 serialization. Required fields are `format` (`tifp-subject-set-v1`), subject/run IDs, creation time, report, dossier, validation artifacts including the external G7 result, requirements verification plan, source manifest, protocol, all instructions, reviewer-control manifest, and manifest hash; every artifact record has ID, canonical path, bytes, and SHA-256. The acyclic order is: source/instructions freeze → reviewer controls freeze → report/dossier/validation freeze → subject set freezes → review context freezes → review occurs → disposition appends → publication record appends. The reviewer receives authorized complete primary sources and exact generated subject artifacts. A canonical disposition records review/subject IDs and hashes, invocation and model disclosure, independence attestations, checks, findings, blockers, advisories, disposition, completion time, and record hash.

- **TIFP-REV-001 — SHALL**: Adversarial review SHALL use a separate invocation from drafting.
- **TIFP-REV-002 — SHALL**: The reviewer invocation SHALL exclude drafting conversation history.
- **TIFP-REV-003 — SHALL**: The reviewer invocation SHALL exclude drafting-agent memory.
- **TIFP-REV-004 — SHALL**: Reviewer context SHALL be rebuilt from the current subject set and immutable source manifest.
- **TIFP-REV-004A — SHALL**: Report, dossier, and validation bytes SHALL enter review only as `bound_generated_artifact` data.
- **TIFP-REV-004B — SHALL**: Reviewer controls SHALL be frozen before subject-set binding.
- **TIFP-REV-004C — SHALL**: The reviewer-control-manifest hash SHALL be in the subject-set manifest.
- **TIFP-REV-004D — SHALL**: Creation of a review-context manifest SHALL NOT modify the frozen source manifest.
- **TIFP-REV-004E — SHALL**: Review context SHALL reference unchanged source-manifest and subject-set hashes.
- **TIFP-REV-004F — SHALL**: The reviewer SHALL receive authorized complete primary sources and exact generated subject artifacts.
- **TIFP-REV-004G — SHALL**: Generated artifact text SHALL NOT be interpreted as reviewer control instructions.
- **TIFP-REV-005 — SHALL**: The reviewer SHALL have no authority to modify the report.
- **TIFP-REV-006 — SHALL**: The reviewer SHALL inspect the current report artifact.
- **TIFP-REV-007 — SHALL**: The reviewer SHALL inspect primary sources.
- **TIFP-REV-008 — SHALL**: The reviewer SHALL inspect quote accuracy.
- **TIFP-REV-009 — SHALL**: The reviewer SHALL inspect timestamp accuracy.
- **TIFP-REV-010 — SHALL**: The reviewer SHALL inspect speech-act accuracy.
- **TIFP-REV-011 — SHALL**: The reviewer SHALL inspect overclaiming.
- **TIFP-REV-012 — SHALL**: The reviewer SHALL inspect contradiction handling.
- **TIFP-REV-013 — SHALL**: The reviewer SHALL inspect counterevidence handling.
- **TIFP-REV-014 — SHALL**: The reviewer SHALL inspect privacy leakage.
- **TIFP-REV-015 — SHALL**: The reviewer SHALL inspect client-facing readability against TIFP-REV-015A.
- **TIFP-REV-015A — SHALL**: Readability SHALL pass only when every section states its main finding or purpose, observations are distinguishable from recommendations and unknowns, acronyms are defined at first use, notes remain client-intelligible without exposing dossier mechanics, internal grading/provenance jargon is absent, and no unresolved prose ambiguity could change a reasonable decision-maker's interpretation.
- **TIFP-REV-015B — SHALL**: A note SHALL count as client-intelligible only when a reviewer who is not shown the dossier or protocol can identify from that note the source type, meeting or document date, and exact timestamp, page, or line locator for the supported claim.
- **TIFP-REV-015C — SHALL**: Internal dossier jargon SHALL include at minimum protocol requirement/gate IDs, artifact-class tokens, hashes, evidence grades, internal source IDs without human-readable locators, and verifier mechanics; any such token in client narrative SHALL fail readability unless the client explicitly requested it.
- **TIFP-REV-015D — SHALL**: Prose ambiguity testing SHALL require two independently assigned reviewers to summarize each material claim's actor, action or condition, timing, scope, and confidence; any substantive discrepancy SHALL be resolved in the report or readability SHALL fail.
- **TIFP-REV-016 — SHALL**: The reviewer SHALL inspect footnote referential integrity.
- **TIFP-REV-017 — SHALL**: The reviewer SHALL inspect package integrity.
- **TIFP-REV-018 — SHALL**: Planned permission to use the same underlying model SHALL be disclosed before S8 in reviewer controls and dossier preparation metadata.
- **TIFP-REV-018A — SHALL**: Actual same-model use at S9 SHALL be disclosed in the immutable review disposition without editing the dossier.
- **TIFP-REV-019 — SHALL**: Use of the same underlying model SHALL occur only through an isolated reviewer invocation.
- **TIFP-REV-020 — SHALL**: A failed reviewer invocation SHALL NOT be recorded as a pass.
- **TIFP-REV-021 — SHALL**: An incomplete review SHALL NOT be recorded as a pass.
- **TIFP-REV-022 — SHALL**: The immutable disposition SHALL be appended without editing any subject artifact.
- **TIFP-REV-023 — SHALL**: A disposition SHALL bind exactly one subject-set-manifest hash.

### 15.1 Post-review audit-control records

A publication decision is canonical JSON at `publication/decisions/publication-decision-<id>.json` with `format` (`tifp-publication-decision-v1`), decision/run/engagement IDs, subject-set and passing review-disposition hashes, frozen engagement-authorization hash, named publication-authority identifier, `publication_decision_approval_evidence_sha256`, evaluated `B`, `R`, `A`, `I`, and `P` values, blocker/advisory inventories, bound advisory-acceptance and advisory-resolution hashes, package-integrity result hash, current requirements-trace head, exact publication payload paths/byte counts/hashes, `created_at`, `prior_control_sha256`, and `record_sha256`, using Section 6.1 rules. The publication-decision approval evidence authorizes only this exact publication decision and payload; it is distinct from authorization-record approval evidence.

An advisory acceptance is canonical JSON at `review/advisory-acceptances/advisory-acceptance-<id>.json` with `format` (`tifp-advisory-acceptance-v1`), record/advisory/run IDs, subject-set hash, review-disposition hash, accepting owner identifier, rationale, disclosure location, `created_at`, `prior_control_sha256`, and `record_sha256`.

An advisory resolution is canonical JSON at `review/advisory-resolutions/advisory-resolution-<id>.json` with `format` (`tifp-advisory-resolution-v1`), record/advisory/run IDs, subject-set and review-disposition hashes, status (`corrected`, `withdrawn_with_evidence`, or `superseded_by_blocker`), resolving protocol-owner identifier, rationale, observed-evidence hashes, replacement blocker ID when applicable, resulting current subject-set/disposition hashes when correction changed reviewed bytes, `created_at`, `prior_control_sha256`, and `record_sha256`, using Section 6.1 rules. A closure-attempt record is canonical JSON at `closure/attempts/closure-attempt-<sequence>-<id>.json` with `format` (`tifp-closure-attempt-v1`), attempt/run IDs, sequence, G11 start/completion times, input audit-control and trace-head hashes, outcome (`pass`, `fail`, `error`, or `incomplete`), output/error evidence hashes, prior-attempt hash or null, and `record_sha256`. A closure record is canonical JSON at `closure/closure-<id>.json` with `format` (`tifp-closure-v1`), closure/run IDs, terminal state, latest available subject-set/disposition/publication hashes or explicit nulls with failure-stage identifiers, successful closure-attempt hash, `resolved_exception_sha256[]`, `remediation_status` (`resolved`), retention/quarantine/access-revocation/disposal actions and schedules, owner, next review date, `created_at`, `prior_control_sha256`, requirements-trace-chain head through entry to S11, and `record_sha256`. A closure-exception record is a permanently immutable historical open snapshot at `closure/exceptions/closure-exception-<id>.json` with `format` (`tifp-closure-exception-v1`), exception/run IDs, `exception_state` (`recorded_open` only), G11 outcome, detected time, latest immutable closure-attempt and audit-control hashes, quarantined publication paths and access-revocation results, owner, remediation, retry due time, retry count, `prior_control_sha256`, and `record_sha256`; it contains no resolving-closure hash or mutable resolution status. Resolution exists exclusively in the later closure record's `resolved_exception_sha256[]`. All use Section 6.1 serialization and self-hash rules. A later trace event may verify a completed control record by referencing its immutable hash; a record never claims to bind an event proving that record's own completed existence.

- **TIFP-CTL-001 — SHALL**: Every advisory acceptance SHALL conform to the advisory-acceptance path, schema, serialization, and self-hash rules.
- **TIFP-CTL-002 — SHALL**: Every advisory acceptance SHALL bind one subject set and its review disposition.
- **TIFP-CTL-003 — SHALL**: Every advisory acceptance SHALL bind the prior audit-control-chain head.
- **TIFP-CTL-004 — SHALL**: A publication record SHALL bind every advisory-acceptance record required by its publication decision.
- **TIFP-CTL-005 — SHALL**: Every closure record SHALL conform to the closure path, schema, serialization, and self-hash rules.
- **TIFP-CTL-006 — SHALL**: Every closure record SHALL bind the latest available run-artifact hashes and prior control-chain head.
- **TIFP-CTL-007 — SHALL**: A denied pre-S8 closure SHALL identify the failed stage and bind the latest frozen context, manifest, and validation hashes available at failure.
