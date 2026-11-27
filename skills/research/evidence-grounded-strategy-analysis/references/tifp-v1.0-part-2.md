- **TIFP-CTL-008 — SHALL**: A closure record SHALL bind the requirements-trace-chain head through completion of all prerequisites for entry to S11.
- **TIFP-CTL-008A — SHALL**: A control record SHALL NOT claim to bind a trace event that verifies that same completed control record.
- **TIFP-CTL-009 — SHALL**: An advisory-acceptance or closure record SHALL be immutable after creation.
- **TIFP-CTL-010 — SHALL**: Every closure-exception record SHALL conform to its canonical path, schema, serialization, predecessor binding, and self-hash rules.
- **TIFP-CTL-011 — SHALL**: Every G11 execution or retry SHALL produce one immutable canonical closure-attempt record.
- **TIFP-CTL-012 — SHALL**: A successful G11 retry SHALL produce a closure record that binds the successful attempt and every resolved closure-exception-record hash.
- **TIFP-CTL-013 — SHALL**: A closure exception SHALL be considered resolved only when the closure record identifies it in `resolved_exception_sha256[]` and records `remediation_status` as `resolved`.
- **TIFP-CTL-014 — SHALL**: Every advisory resolution SHALL conform to its canonical path, schema, serialization, predecessor binding, and self-hash rules.
- **TIFP-CTL-015 — SHALL**: Only the protocol owner SHALL create an advisory-resolution record.
- **TIFP-CTL-016 — SHALL**: A corrected advisory that changes reviewed bytes SHALL require a fresh subject set and passing review before it counts as resolved.
- **TIFP-CTL-017 — SHALL**: An advisory SHALL count as resolved only when one current valid advisory-resolution record identifies its advisory ID and all status-specific evidence fields pass validation.
- **TIFP-CTL-018 — SHALL**: The publication record SHALL bind one valid acceptance or resolution record for every recorded advisory.

## 16. Artifact-set binding and invalidation

- **TIFP-BND-001 — SHALL**: A review SHALL bind the report hash.
- **TIFP-BND-002 — SHALL**: A review SHALL bind the dossier hash.
- **TIFP-BND-003 — SHALL**: A review SHALL bind the source-manifest hash.
- **TIFP-BND-004 — SHALL**: A review SHALL bind the protocol hash and version.
- **TIFP-BND-005 — SHALL**: A review SHALL bind every current-engagement instruction hash.
- **TIFP-BND-005A — SHALL**: A review SHALL bind the pre-frozen reviewer-control-manifest hash and every subject validation hash.
- **TIFP-BND-005B — SHALL**: A review SHALL bind the frozen requirements-verification-plan hash.
- **TIFP-BND-006 — SHALL**: A change to the report SHALL invalidate the prior passing review.
- **TIFP-BND-007 — SHALL**: A change to the dossier SHALL invalidate the prior passing review.
- **TIFP-BND-008 — SHALL**: A change to the source manifest SHALL invalidate the prior passing review.
- **TIFP-BND-009 — SHALL**: A change to the protocol SHALL invalidate the prior passing review.
- **TIFP-BND-010 — SHALL**: A change to a current-engagement instruction SHALL invalidate the prior passing review.
- **TIFP-BND-011 — SHALL**: Removal of a dossier entry SHALL invalidate each affected footnote.
- **TIFP-BND-012 — SHALL**: Renaming of a dossier entry SHALL invalidate each affected footnote.
- **TIFP-BND-013 — SHALL**: A change to a dossier source location SHALL invalidate each affected footnote.
- **TIFP-BND-014 — SHALL**: A change to referenced evidence content SHALL invalidate each affected footnote.
- **TIFP-BND-015 — SHALL**: An invalidated artifact set SHALL receive fresh validation before publication.
- **TIFP-BND-016 — SHALL**: An invalidated artifact set SHALL receive fresh independent review before publication.

## 17. Blockers, advisories, and publication truth table

### 17.1 Non-waivable blockers

- **TIFP-BLK-001 — SHALL**: Privacy leakage SHALL be a non-waivable blocker.
- **TIFP-BLK-002 — SHALL**: Outside-root context SHALL be a non-waivable blocker.
- **TIFP-BLK-003 — SHALL**: Unmanifested context SHALL be a non-waivable blocker.
- **TIFP-BLK-004 — SHALL**: A hash mismatch SHALL be a non-waivable blocker.
- **TIFP-BLK-005 — SHALL**: An unresolved footnote SHALL be a non-waivable blocker.
- **TIFP-BLK-006 — SHALL**: A stale footnote SHALL be a non-waivable blocker.
- **TIFP-BLK-007 — SHALL**: An inaccurate quotation SHALL be a non-waivable blocker.
- **TIFP-BLK-008 — SHALL**: An invented source location SHALL be a non-waivable blocker.
- **TIFP-BLK-009 — SHALL**: Failed package integrity SHALL be a non-waivable blocker.
- **TIFP-BLK-010 — SHALL**: A review not bound to the current artifact set SHALL be a non-waivable blocker.
- **TIFP-BLK-010A — SHALL**: Every failed, errored, or incomplete mandatory gate that is due at the decision stage SHALL be a non-waivable blocker.
- **TIFP-BLK-010H — SHALL**: G11 SHALL NOT be treated as due during the S10 publication decision.
- **TIFP-BLK-010B — SHALL**: Every applicable failed privacy, provenance, or isolation requirement SHALL be a non-waivable blocker.
- **TIFP-BLK-010C — SHALL**: Every applicable failed quote accuracy or quote-selection requirement SHALL be a non-waivable blocker.
- **TIFP-BLK-010D — SHALL**: Every applicable failed citation, source-location, or client-note-rendering requirement SHALL be a non-waivable blocker.
- **TIFP-BLK-010E — SHALL**: Every applicable failed artifact, package, or referential-integrity requirement SHALL be a non-waivable blocker.
- **TIFP-BLK-010F — SHALL**: Every applicable failed current-review requirement SHALL be a non-waivable blocker.
- **TIFP-BLK-010G — SHALL**: Any ambient-context contribution SHALL be a non-waivable blocker.
- **TIFP-BLK-011 — SHALL**: Disclosure SHALL NOT waive a blocker.
- **TIFP-BLK-012 — SHALL**: No role SHALL have authority to waive a blocker.

### 17.2 Advisories

- **TIFP-ADV-001 — SHALL**: Only a style preference or disclosed methodological limitation SHALL be eligible for advisory treatment.
- **TIFP-ADV-001A — SHALL**: An inaccessible package region SHALL NOT be eligible for advisory treatment.
- **TIFP-ADV-002 — SHALL**: Only the protocol owner SHALL accept an advisory finding.
- **TIFP-ADV-003 — SHALL**: Each accepted advisory SHALL have a stable advisory ID.
- **TIFP-ADV-004 — SHALL**: Each accepted advisory SHALL record the protocol owner’s identity or role identifier.
- **TIFP-ADV-005 — SHALL**: Each accepted advisory SHALL record its rationale.
- **TIFP-ADV-006 — SHALL**: Each accepted advisory SHALL record its disclosure location.
- **TIFP-ADV-007 — SHALL**: An open question that is not a blocker SHALL be disclosed.
- **TIFP-ADV-008 — SHALL**: A nonblocking limitation SHALL be disclosed.

### 17.3 Publication truth table

At a decision stage, a gate or requirement is **due** when its protocol stage is at or before that decision stage and its applicability trigger is present. `B = 1` iff any due mandatory gate failed, errored, or is incomplete, or any due applicable privacy, provenance/isolation, quote accuracy/selection, citation/location/note rendering, artifact/package/referential integrity, ambient-context, or current-review requirement failed, errored, or is incomplete; otherwise `B = 0`. A future gate is not incomplete before its stage begins, but it remains mandatory when due. `R` = passing immutable review bound to the current subject set; `A` = every recorded advisory resolved or accepted; `I` = package and referential integrity pass; `P` = the frozen authorization record's named publication authority approved this exact decision.

| B | R | A | I | P | Publish |
|---:|---:|---:|---:|---:|:---|
| 1 | * | * | * | * | NO |
| 0 | 0 | * | * | * | NO |
| 0 | 1 | 0 | * | * | NO |
| 0 | 1 | 1 | 0 | * | NO |
| 0 | 1 | 1 | 1 | 0 | NO |
| 0 | 1 | 1 | 1 | 1 | YES |

- **TIFP-PUB-001 — SHALL**: Publication SHALL follow the truth table.
- **TIFP-PUB-002 — SHALL**: Publication SHALL occur only when no blocker is open.
- **TIFP-PUB-003 — SHALL**: Publication SHALL occur only with a passing review bound to the current artifact set.
- **TIFP-PUB-004 — SHALL**: Publication SHALL occur only when every recorded advisory has one current valid resolution or acceptance record.
- **TIFP-PUB-005 — SHALL**: Publication SHALL occur only when package and referential integrity pass.
- **TIFP-PUB-006 — SHALL**: The publication decision SHALL be an immutable append-only publication record outside the dossier.
- **TIFP-PUB-007 — SHALL**: The publication record SHALL bind the subject-set and passing review-disposition hashes without editing reviewed artifacts.
- **TIFP-PUB-008 — SHALL**: The publication record SHALL bind every required advisory-acceptance or advisory-resolution record hash and the current requirements-trace-chain head.
- **TIFP-PUB-009 — SHALL**: The publication record SHALL bind the frozen engagement-authorization hash and the named publication authority's approval evidence.
- **TIFP-PUB-010 — SHALL**: Every publication record SHALL conform to its canonical path, schema, serialization, predecessor binding, and self-hash rules.
- **TIFP-PUB-011 — SHALL**: A publication record SHALL bind the exact approved publication payload paths, byte counts, and hashes.

## 18. Failure handling

- **TIFP-FAIL-001 — SHALL**: An outside-root input SHALL terminate the run before model invocation.
- **TIFP-FAIL-002 — SHALL**: An in-root unmanifested file selected for context SHALL terminate the run before model invocation.
- **TIFP-FAIL-003 — SHALL**: A manifest hash mismatch SHALL terminate the run before model invocation.
- **TIFP-FAIL-004 — SHALL**: A prohibited identifier in a context-bound file SHALL terminate the run before model invocation.
- **TIFP-FAIL-005 — SHALL**: An ambient-channel contribution SHALL terminate context building before model invocation.
- **TIFP-FAIL-006 — SHALL**: A context failure SHALL record the stage.
- **TIFP-FAIL-007 — SHALL**: A context failure SHALL record the affected artifact identifier without copying prohibited content.
- **TIFP-FAIL-008 — SHALL**: A context failure SHALL record the failed rule ID.
- **TIFP-FAIL-009 — SHALL**: A post-invocation evidence defect SHALL quarantine all downstream artifacts from that invocation.
- **TIFP-FAIL-010 — SHALL**: A validation failure SHALL prevent binding a passing artifact set.
- **TIFP-FAIL-011 — SHALL**: A review failure SHALL prevent publication.
- **TIFP-FAIL-012 — SHALL**: A material edit after review SHALL return the workflow to validation.
- **TIFP-FAIL-013 — SHALL**: A technical error SHALL be recorded as an error rather than a substantive pass or fail.
- **TIFP-FAIL-014 — SHALL**: Recovery after a pre-invocation isolation failure SHALL begin with a newly validated context build.

## 19. Package privacy and integrity inspection

- **TIFP-PKG-001 — SHALL**: Privacy scanning SHALL inspect document body text.
- **TIFP-PKG-002 — SHALL**: Privacy scanning SHALL inspect tables.
- **TIFP-PKG-003 — SHALL**: Privacy scanning SHALL inspect document metadata.
- **TIFP-PKG-004 — SHALL**: Privacy scanning SHALL inspect headers.
- **TIFP-PKG-005 — SHALL**: Privacy scanning SHALL inspect footers.
- **TIFP-PKG-006 — SHALL**: Privacy scanning SHALL inspect comments.
- **TIFP-PKG-007 — SHALL**: Privacy scanning SHALL inspect links and relationships.
- **TIFP-PKG-008 — SHALL**: Privacy scanning SHALL inspect embedded text where technically accessible.
- **TIFP-PKG-009 — SHALL**: Privacy scanning SHALL inspect embedded objects where technically accessible.
- **TIFP-PKG-010 — SHALL**: Technically inaccessible package regions SHALL be recorded as validation limitations.
- **TIFP-PKG-010A — SHALL**: Every technically inaccessible package region SHALL be a non-waivable package-integrity and privacy blocker.
- **TIFP-PKG-010B — SHALL**: A package containing a technically inaccessible region SHALL NOT pass G7 or publication.
- **TIFP-PKG-011 — SHALL**: A package-format parser failure SHALL be a package-integrity blocker.

## 20. Acceptance tests

An implementation conforms only when it records pass/fail evidence for every test below against the versioned implementation and fixture.

| Test ID | Stimulus | Expected result |
|---|---|---|
| AT-ISO-001 | Select a file outside the engagement root. | Run terminates before invocation. |
| AT-ISO-002 | Select an in-root file absent from the manifest. | Run terminates before invocation and records the exception. |
| AT-ISO-003 | Alter one byte of a manifest-listed file. | Run terminates before invocation. |
| AT-ISO-004 | Insert a prohibited identifier into a context-bound file. | Run terminates before invocation. |
| AT-ISO-005 | Enable conversation history, memory, prior output, retrieval, template, tool result, or auto-discovered file. | Context build fails. |
| AT-ISO-006 | Inspect serialized payload and context manifest. | Every payload byte is covered exactly once by an authorized segment. |
| AT-ISO-007 | Remove positive provenance but leave identifier scan clean. | Context build fails. |
| AT-ISO-008 | Put other-engagement metadata, attachment, relationship, embedded object, or fact with no prohibited name anywhere under the root. | Ingestion fails and root is quarantined. |
| AT-ISO-009 | Supply ambiguous origin or missing human attestation with a clean scan. | Ingestion fails closed. |
| AT-SCM-001 | Authorize primary sources plus exact current-run class-5 report/dossier bytes at allowed paths for S9. | Stage and outbound manifests pass. |
| AT-SCM-002 | Launder report as instruction, omit source-manifest reference, alter hash, use earlier-run output, or use class 5 at a wrong path/stage. | Context build fails. |
| AT-SCM-003 | Freeze controls before binding and include their hash in subject set. | Acyclic order passes; post-bind control creation/change fails. |
| AT-SCM-004 | Omit an authorized primary source needed to verify a claim, substitute a generated summary, or omit/alter an exact subject artifact. | Review-context validation fails before reviewer invocation. |
| AT-EVD-001 | Add a material assertion without evidence and without assumption/recommendation marking. | Validation fails. |
| AT-EVD-002 | Use a real timestamp/page label. | Location validation passes. |
| AT-EVD-003 | Invent a timestamp/page label. | Validation fails. |
| AT-EVD-004 | Present competing end states as consensus. | Analysis review fails. |
| AT-EVD-005 | Omit material counterevidence. | Analysis review fails. |
| AT-EVD-006 | Keep only a target sentence while adjacent turns change referent, negation, qualification, condition, speech act, or contradiction. | Context-sufficiency validation fails. |
| AT-EVD-007 | Capture adjacent turns, expand through all meaning-changing context, and record rationale. | Check passes only after independent source verification. |
| AT-DEP-001 | Verify a summary against its parent transcript. | Summary remains derivative and non-independent. |
| AT-DEP-002 | Add two derivatives of one transcript. | Corroborating lineage count remains one. |
| AT-QTE-001 | Apply NFC, quote-mark, line-break, or repeated-whitespace normalization only. | Quote validation passes. |
| AT-QTE-002 | Substitute a lexical word. | Quote validation fails. |
| AT-QTE-003 | Change a number. | Quote validation fails. |
| AT-QTE-004 | Change negation. | Quote validation fails. |
| AT-QTE-005 | Reorder words. | Quote validation fails. |
| AT-QTE-006 | Remove a disfluency without ellipsis. | Quote validation fails. |
| AT-QTE-007 | Use an ellipsis without preserved omitted spans. | Quote validation fails. |
| AT-QTE-008 | Put a paraphrase in quotation marks. | Quote validation fails. |
| AT-QTE-009 | Attribute unreliable diarization to a named speaker. | Quote validation fails. |
| AT-QTE-010 | Use authenticated or human-verified attribution with no conflict. | Reliability check passes. |
| AT-RPT-001 | Insert an inline grade, verdict, corpus ID, raw path, model/provider name, or verifier mechanics. | Report validation fails. |
| AT-RPT-002 | Render every note with source label and real timestamp/page/line span, optionally a concise validated quote. | Check passes; an opaque ID alone fails. |
| AT-RPT-003 | Use a quote only to signal evidence or omit drafter rationale/reviewer acceptance. | Quote-selection review fails; paraphrase with locator note is required. |
| AT-DOS-001 | Remove a required dossier section. | Dossier validation fails. |
| AT-REF-001 | Remove or rename a referenced dossier entry. | Footnote and review become invalid. |
| AT-REF-002 | Change a referenced source location. | Footnote and review become invalid. |
| AT-BND-001 | Change the report hash after review. | Review becomes invalid. |
| AT-BND-002 | Change the dossier hash after review. | Review becomes invalid. |
| AT-BND-003 | Change the source manifest, protocol, or instruction hash after review. | Review becomes invalid. |
| AT-BND-004 | Append disposition/publication records without changing subject artifacts. | Binding remains valid and lifecycle terminates. |
| AT-BND-005 | Add disposition/publication data to dossier after binding. | Subject hash fails; publication denied. |
| AT-REV-001 | Supply drafting history or memory to reviewer invocation. | Independence check fails. |
| AT-REV-002 | Give reviewer report-modification authority. | Independence check fails. |
| AT-REV-003 | Present prose with undefined acronyms, indistinguishable recommendation/observation status, dossier jargon, or decision-changing ambiguity. | TIFP-REV-015A fails and publication is denied. |
| AT-RSP-001 | Return model bytes but omit the response record or transformation lineage. | Stage gate fails and generated artifacts are denied. |
| AT-RSP-002 | Return an error or partial completion and mark it complete. | Response validation fails and downstream use is denied. |
| AT-PKG-001 | Seed residue in body, table, metadata, header, footer, comment, relationship, or accessible embedded text. | Privacy validation fails. |
| AT-PKG-002 | Include an embedded or package region the validator cannot inspect. | G7 fails, `B = 1`, advisory treatment is rejected, and publication is denied. |
| AT-TRC-001 | Freeze a complete requirements verification plan before S8 and include its hash in the subject set. | Binding and review-context validation pass without claiming future-stage results. |
| AT-TRC-002 | Omit, replace, or edit the verification plan after binding. | Subject-set or hash validation fails and publication is denied. |
| AT-TRC-003 | Append stage-appropriate immutable trace events before and after S8. | Each event binds its predecessor; future-stage requirements remain `pending`; subject bytes do not change. |
| AT-TRC-004 | Claim publication while a publication-prerequisite result is missing, pending, failed, or errored. | Publication is denied. |
| AT-TRC-005 | Claim conformance before S11 or with any applicable result missing, pending, failed, or errored. | Conformance claim is denied. |
| AT-CTL-001 | Accept a post-review advisory and publish. | Canonical acceptance binds advisory, disposition, subject set, and predecessor; the later publication record binds the acceptance hash. |
| AT-CTL-002 | Close a failed pre-S8 run. | Canonical closure binds failure stage, latest available frozen hashes, trace-chain head, and retention actions without publication. |
| AT-CLS-001 | Fail, error, or leave incomplete G3, G7, G8, or G9. | The next productive stage is denied; control transitions to S11 Close-Denied and executes or schedules required retention/quarantine actions without publication. |
| AT-CLS-002 | Fail, error, or leave incomplete G11. | Run enters nonterminal `closure_exception`; an immutable exception record is appended; publication access is quarantined; no S11-to-S11 transition or conformance occurs; G11 is retried after remediation. |
| AT-CLS-003 | Retry G11 successfully after a recorded failure. | New attempt and closure records bind the failed attempt and exception; the latest effective G11 result is pass; history remains immutable; the resolved failure no longer blocks conformance. |
| AT-PUB-001 | Open any non-waivable blocker and disclose it. | Publication remains denied. |
| AT-PUB-002 | Resolve blockers but use stale review. | Publication remains denied. |
| AT-PUB-003 | No blockers; current passing review; advisories resolved/accepted; integrity passes; named publication authority approves. | Publication is authorized. |
| AT-PUB-004 | Evaluate S10 while G11 has not begun. | G11 is future, not incomplete; it does not set `B = 1` at S10. |
| AT-PUB-005 | B=0, R=1, A=1, I=1, but named publication-authority approval is absent or mismatched. | P=0 and publication is denied. |
| AT-ADV-001 | Mark an advisory “resolved” without a canonical resolution record. | A=0 and publication is denied. |
| AT-QSL-001 | Select a quote without one enumerated material-improvement condition or without a paraphrase comparison. | Quote selection fails and paraphrase is required. |
| AT-CLS-004 | Declare an unlisted or compound artifact class. | Artifact validation fails. |
| AT-SCP-001 | Complete S1 without copying a complete primary transcript-text source, or attempt S3 before S2 hash-pins it. | Run is denied before evidence extraction. |
| AT-REV-004 | Give two isolated reviewers the client report but not the dossier/protocol. | Every note yields the same source type/date/locator and every material claim summary agrees on actor, action/condition, timing, scope, and confidence; otherwise readability fails. |

- **TIFP-AT-001 — SHALL**: Every listed acceptance test SHALL be executed before an implementation claims v1.0 conformance.
- **TIFP-AT-002 — SHALL**: Each acceptance-test result SHALL identify the tested implementation version.
- **TIFP-AT-003 — SHALL**: Each acceptance-test result SHALL preserve actual observed output.
- **TIFP-AT-004 — SHALL**: A failed mandatory acceptance test SHALL prevent a conformance claim.
- **TIFP-AT-005 — SHALL**: A skipped mandatory acceptance test SHALL prevent a conformance claim.

Acceptance-test results are implementation-conformance artifacts outside engagement workspaces. Each result is canonical JSON at `protocol-conformance/acceptance-tests/<implementation-id>/results/<test-id>.json` with `format` (`tifp-acceptance-test-result-v1`), protocol version/hash, implementation ID/version/hash, test ID, fixture/input hashes, execution command or method, start/completion times, observed stdout/stderr or equivalent observed-output hashes, outcome (`pass`, `fail`, `error`, or `skipped`), verifier, and `record_sha256`. The canonical manifest at `protocol-conformance/acceptance-tests/<implementation-id>/acceptance-test-manifest.json` lists every required test ID, result path, byte count, and hash; records the complete/missing set; binds protocol and implementation hashes; and self-hashes under Section 6.1.

- **TIFP-AT-006 — SHALL**: Every acceptance-test result SHALL conform to its canonical path, schema, serialization, and self-hash rules.
- **TIFP-AT-007 — SHALL**: The acceptance-test manifest SHALL enumerate every acceptance test listed in Section 20 exactly once.
- **TIFP-AT-008 — SHALL**: The acceptance-test manifest SHALL bind every result path, byte count, and hash plus the tested protocol and implementation hashes.
- **TIFP-AT-009 — SHALL**: An implementation conformance claim SHALL cite the exact acceptance-test-manifest hash.
- **TIFP-AT-010 — SHALL**: Acceptance-test results and their manifest SHALL NOT enter an engagement evidence set or engagement model context.

## 21. Minimal synthetic acceptance-test fixture

This fixture is fictional and contains no real engagement data.

```text
fixture/engagement-root/
  isolation-policy.json
  source-manifest.json
  instructions/INSTR-001.md
  instructions/engagement-authorization-payload.json
  instructions/approvals/authorization-record-approval.json
  instructions/engagement-authorization.json
  evidence/primary/SRC-000001.txt
  evidence/derivative/SRC-000002.md
  evidence/unmanifested.txt
fixture/outside-root/SRC-OUTSIDE.txt
```

`SRC-000001.txt`:

```text
[00:00:10] Participant 1: We can test one queue first.
[00:00:14] Participant 2: That would not prove adoption across every team.
[00:00:20] Participant 1: Is a six-week test feasible?
[00:00:26] Participant 2: We have not approved a timeline.
```

`SRC-000002.md`:

```text
The participants discussed a narrow queue test. This summary is derived only from SRC-000001.
```

`INSTR-001.md`:

```text
Prepare initial findings from the frozen synthetic evidence packet under TIFP v1.0.
```

The three authorization artifacts conform to Section 3 using only synthetic role, purpose, audience, and approval IDs; their acyclic payload→approval→final-record hashes are computed after fixture materialization. `isolation-policy.json` uses synthetic engagement key `ENG-SYN-001`, allows only fixture identifiers, and includes prohibited sentinel `PROHIBITED-CARRYOVER-TOKEN`. At S2, `source-manifest.json` lists only the six intended source/instruction files and their computed hashes; it intentionally omits `evidence/unmanifested.txt`. Test runners compute fixture hashes after materialization rather than copying illustrative hashes from this document.

Expected valid finding: a narrow queue test was discussed. Expected limitation: it cannot establish adoption across every team. Expected unknown: no timeline was approved. Expected speech-act result: the six-week statement is a question, not a decision. Expected dependence result: the summary does not independently corroborate the transcript.

- **TIFP-FIX-001 — SHALL**: The conformance fixture SHALL contain only synthetic content.
- **TIFP-FIX-002 — SHALL**: The fixture manifest SHALL be frozen before its valid-path test.
- **TIFP-FIX-003 — SHALL**: The fixture SHALL include an outside-root negative input.
- **TIFP-FIX-004 — SHALL**: The fixture SHALL include an in-root unmanifested negative input.
- **TIFP-FIX-005 — SHALL**: The fixture SHALL include a prohibited-identifier negative input.
- **TIFP-FIX-006 — SHALL**: The fixture SHALL exercise source dependence.
- **TIFP-FIX-007 — SHALL**: The fixture SHALL exercise speech-act classification.
- **TIFP-FIX-008 — SHALL**: The fixture SHALL exercise pilot-scope limitation.
- **TIFP-FIX-009 — SHALL**: The fixture SHALL include the canonical synthetic authorization payload, approval evidence, and final authorization record and SHALL manifest all three at S2.

## 22. Retention, disposal, and closure

TIFP sets no universal retention period. The engagement owner's frozen instruction determines lawful and contractual retention, archival, legal hold, quarantine, access revocation, and disposal.

- **TIFP-RET-001 — SHALL**: Each run SHALL include an owner-approved retention/disposal plan as a frozen current-engagement instruction.
- **TIFP-RET-002 — SHALL**: The plan SHALL define triggers and periods for sources, work, deliverables, review controls, publication records, and backups.
- **TIFP-RET-003 — SHALL**: The plan SHALL define archival, legal-hold, access-revocation, quarantine, and disposal procedures.
- **TIFP-RET-004 — SHALL**: S11 SHALL execute due actions or record the schedule, owner, and next review date.
- **TIFP-RET-005 — SHALL**: Disposal completion SHALL be recorded without copying disposed sensitive content.
- **TIFP-RET-006 — SHALL**: This protocol SHALL NOT assert a universal retention period.

## 23. Versioning and change control

- **TIFP-VER-001 — SHALL**: The protocol SHALL use semantic versioning.
- **TIFP-VER-002 — SHALL**: A privacy-boundary change SHALL increment the major version.
- **TIFP-VER-003 — SHALL**: A schema-breaking change SHALL increment the major version.
- **TIFP-VER-004 — SHALL**: A new backward-compatible requirement SHALL increment the minor version.
- **TIFP-VER-005 — SHALL**: A non-substantive clarification SHALL increment the patch version.
- **TIFP-VER-006 — SHALL**: Every released protocol version SHALL have a SHA-256 hash.
- **TIFP-VER-007 — SHALL**: Every released protocol version SHALL have a change log.
- **TIFP-VER-008 — SHALL**: Every run SHALL record the exact protocol version.
- **TIFP-VER-009 — SHALL**: Every run SHALL record the exact protocol hash.
- **TIFP-VER-010 — SHALL**: A protocol change during a run SHALL create a new bound artifact set.
- **TIFP-VER-011 — SHALL**: A protocol change during a run SHALL require fresh validation.
- **TIFP-VER-012 — SHALL**: A protocol change during a run SHALL require fresh review.
- **TIFP-VER-013 — SHALL**: Every released version's authoritative metadata SHALL identify the protocol owner role and protocol release authority.
- **TIFP-VER-014 — SHALL**: Canonical `protocol-development/RELEASE.json` SHALL bind the exact protocol path, byte count, hash, stakeholder-review hash, independent-release-review hash, Elicit challenge-report ID/path/hash, release authority, release time, and its own `record_sha256`.
- **TIFP-VER-015 — SHALL**: No protocol version SHALL be represented as released unless its exact hash appears in a valid release record.
- **TIFP-VER-016 — SHALL**: Creating the external release record SHALL NOT mutate the protocol bytes it releases.

### 23.1 Protocol-development stakeholder governance

Core stakeholder groups are consulting analyst/drafter, client decision-maker, evidence/privacy steward, independent reviewer, publication authority, records/retention owner, and implementation operator. Canonical `protocol-development/stakeholder-review.json` uses `format` (`tifp-stakeholder-review-v1`), candidate protocol version/path/byte count/hash, one entry per core group with role identifier, stated needs, reviewer/representative role identifier, review outcome (`agree`, `agree_with_advisory`, or `block`), rationale, evidence references, unresolved concerns, review time, protocol-owner disposition, and `record_sha256` under Section 6.1.

- **TIFP-GOV-001 — SHALL**: Every release candidate SHALL have one canonical stakeholder-review record before release review closes.
- **TIFP-GOV-002 — SHALL**: The stakeholder-review record SHALL contain every core stakeholder group exactly once.
- **TIFP-GOV-003 — SHALL**: Each core group SHALL have an identified representative role, stated needs, outcome, rationale, and evidence reference.
- **TIFP-GOV-004 — SHALL**: A `block` outcome or unresolved need affecting privacy, evidence integrity, client usability, reviewability, publication, retention, or implementability SHALL prevent protocol release.
- **TIFP-GOV-005 — SHALL**: The protocol owner SHALL disposition every `agree_with_advisory` outcome before release.
- **TIFP-GOV-006 — SHALL**: A protocol change after stakeholder review SHALL invalidate that record and require fresh stakeholder review.

## 24. Elicit invocation and reference instructions

Elicit may be used either (a) inside an engagement run at S5 Analysis under Sections 4–6, or (b) outside S0–S11 for client-neutral protocol-development challenge. The invocation SHALL declare `lifecycle_scope` and an enumerated stage token. It references this protocol by version and hash; it does not embed engagement evidence into this reusable file.

- **TIFP-ELI-001 — SHALL**: An Elicit invocation SHALL load the approved protocol explicitly by hash.
- **TIFP-ELI-002 — SHALL**: An engagement-run Elicit invocation SHALL use only the five allowlisted classes under their stage restrictions.
- **TIFP-ELI-003 — SHALL**: An Elicit invocation SHALL have its own outbound context manifest.
- **TIFP-ELI-004 — SHALL**: An Elicit output SHALL be treated as a run artifact rather than certification.
- **TIFP-ELI-005 — SHALL**: An engagement-run Elicit challenge finding SHALL enter the dossier with its disposition.
- **TIFP-ELI-006 — SHALL**: An Elicit runtime failure SHALL be recorded as a failed or incomplete invocation.
- **TIFP-ELI-007 — SHALL**: An Elicit runtime failure SHALL NOT be represented as a passed review.
- **TIFP-ELI-008 — SHALL**: Elicit code SHALL NOT be modified as part of applying this protocol.
- **TIFP-ELI-009 — SHALL**: Engagement material SHALL NOT be uploaded unless a separately approved control regime is added by a future protocol version.
- **TIFP-ELI-010 — SHALL**: An engagement-run Elicit analytic challenge SHALL declare `lifecycle_scope` as `engagement_run` and `stage` as `S5_analysis_challenge`.
- **TIFP-ELI-011 — SHALL**: A protocol-development Elicit challenge SHALL declare `lifecycle_scope` as `protocol_development` and `stage` as `external_protocol_challenge`.
- **TIFP-ELI-012 — SHALL**: `external_protocol_challenge` SHALL be outside S0–S11 and SHALL NOT authorize any engagement evidence, instructions, generated artifacts, or dossier disposition.
- **TIFP-ELI-013 — SHALL**: A protocol-development challenge SHALL receive no inputs other than the hash-pinned protocol and hash-pinned client-neutral governing requirements through an isolated outbound manifest.
- **TIFP-ELI-014 — SHALL**: Protocol-development challenge output SHALL be stored as development challenge evidence and SHALL NOT be represented as engagement evidence, review, or certification.

Reference invocation record:

```json
{
  "lifecycle_scope": "protocol_development",
  "stage": "external_protocol_challenge",
  "protocol": {"id": "TIFP", "version": "1.0.0", "sha256": "<candidate-protocol-hash>"},
  "neutral_requirements_manifest_sha256": "<client-neutral-governing-manifest-hash>",
  "outbound_context_manifest": "<context-manifest-path>",
  "ambient_channels": "disabled",
  "expected_output": "challenge findings with evidence links and dispositions"
}
```

## 25. Provenance and v1.0 change note

An earlier Elicit deliberation **FAILED** because requirements-specialist and specification-editor runtime calls failed; it was not certification and is not represented as a pass. Successful Elicit challenge run `doc-20260820-001413-f7b5b1` produced five qualified and two disputed findings; it is development challenge evidence, not certification. RC2 added the main privacy, review, and artifact controls. RC3 removed the G7 dossier cycle, added failure-to-closure transitions, blocked inaccessible package regions, and defined the broader audit package. RC4 separated the pre-S8 verification plan from the append-only requirements-result chain, aligned the governing five-class context boundary, and canonically bound advisory-acceptance and closure records. RC5 added observable readability criteria, stage-specific S7/S9 generated-artifact binding, generic self-hash semantics, explicit error blocking, timed same-model disclosure, independent-review-role semantics, and model-response capture. RC6 removed terminal conformance-claim recursion. RC7 corrected advisory/publication binding direction, publication-decision bootstrapping, incomplete-gate closure, and outbound-context-manifest self-hashing. RC8 removed the G11 self-transition. RC9 defined immutable closure attempts, deterministic remediation resolution, effective-result supersession, and stage-qualified blocker evaluation. RC10 removes the last closure-exception/closure-record circular hash option.

**v1.0 final-candidate change log:** retained RC14 controls; made transcript text mandatory; required audience-and-purpose pair authorization; canonically preserved and access-logged raw response bytes; completed protocol-development context manifests; established model-service approval thresholds; added stakeholder governance; and defined external immutable release authorization.

## 26. Definition of done

- **TIFP-DONE-001 — SHALL**: A conforming run SHALL produce both required deliverables.
- **TIFP-DONE-002 — SHALL**: A conforming run SHALL preserve a complete dossier.
- **TIFP-DONE-003 — SHALL**: A conforming run SHALL pass every applicable gate.
- **TIFP-DONE-004 — SHALL**: A conforming run SHALL pass every mandatory acceptance test.
- **TIFP-DONE-005 — SHALL**: A conforming run SHALL have a current passing independent review.
- **TIFP-DONE-006 — SHALL**: A conforming run SHALL satisfy the publication truth table.
- **TIFP-DONE-007 — SHALL**: An external conformance claim SHALL identify the exact protocol version and hash.

## 27. Requirements verification plan and append-only trace

The canonical pre-S8 plan is `review/requirements-trace/requirements-verification-plan.json`, using Section 6.1 canonical JSON and self-hash rules. It contains `format` (`tifp-requirements-verification-plan-v1`), run/engagement IDs, creation time, frozen state, protocol version/hash, one `plan_records[]` entry per requirement, and `plan_sha256`. Each plan record contains requirement ID, requirement rationale, owner, verification role, method, source trace, applicable stage, applicability trigger, and acceptance-test ID when available; it contains no claimed result.

Results are immutable events at `review/requirements-trace/trace-event-<sequence>-<id>.json`. Each event contains `format` (`tifp-requirements-trace-event-v1`), event/run/engagement IDs, sequence, requirement ID, stage, result (`pass`, `fail`, `error`, `not_applicable`, or `pending`), rationale, observed-evidence references and hashes, verifier, verification time, subject-set hash when available, `supersedes_event_sha256` and `resolution_record_sha256` when resolving a prior outcome, `prior_event_sha256`, and `event_sha256`. `pending` is valid only when the requirement's applicable stage has not begun; it never counts as a pass. The review disposition binds the chain head after review checks but before disposition creation; the publication record binds the chain head after all publication prerequisites but before publication-record creation; and the closure record binds the chain head through entry to S11. Later trace events verify each completed run control record by its immutable hash. Publication requires every prerequisite to be `pass` or validly `not_applicable`. A conformance claim made after S11 is external presentation metadata: it references the closure-record hash and the already-frozen final post-closure run trace head, but is not added to the run package or trace. The verification plan marks `TIFP-DONE-007`, `TIFP-TRC-013`, `TIFP-TRC-015`, and `TIFP-TRC-016` with applicable stage `external_claim`; their presentation-time checks do not extend the run trace.

- **TIFP-TRC-001 — SHALL**: The verification plan SHALL contain exactly one plan record for every requirement ID.
- **TIFP-TRC-002 — SHALL**: Every plan record SHALL identify requirement rationale, owner, verification role, method, source trace, applicable stage, and applicability trigger.
- **TIFP-TRC-003 — SHALL**: The verification plan SHALL conform to its path, schema, serialization, and self-hash rules.
- **TIFP-TRC-004 — SHALL**: The verification plan SHALL freeze before S8.
- **TIFP-TRC-005 — SHALL**: The subject-set manifest SHALL bind the exact verification-plan path, byte count, and hash.
- **TIFP-TRC-006 — SHALL**: Each evaluated run-stage requirement SHALL produce one immutable trace event containing observed evidence and result.
- **TIFP-TRC-006A — SHALL**: External-claim presentation checks SHALL NOT append a run trace event or mutate the frozen run audit package.
- **TIFP-TRC-007 — SHALL**: Each trace event SHALL conform to its path, schema, serialization, self-hash, sequence, and predecessor-hash rules.
- **TIFP-TRC-008 — SHALL**: `pending` SHALL be used only before the requirement's applicable stage begins.
- **TIFP-TRC-009 — SHALL**: `pending`, missing metadata, missing evidence, `fail`, or `error` SHALL NOT count as a pass.
- **TIFP-TRC-010 — SHALL**: `not_applicable` SHALL require an absent stated trigger and verification-lead rationale.
- **TIFP-TRC-011 — SHALL**: The review disposition SHALL bind the requirements-trace-chain head through completion of review checks and before disposition creation.
- **TIFP-TRC-012 — SHALL**: The publication record SHALL bind the requirements-trace-chain head through completion of publication prerequisites and before publication-record creation.
- **TIFP-TRC-013 — SHALL**: The external conformance claim SHALL reference the immutable closure-record hash and already-frozen final post-closure run trace-chain head.
- **TIFP-TRC-014 — SHALL**: Publication SHALL be denied when the effective result of any due publication prerequisite is missing, pending, failed, errored, or invalidly not applicable.
- **TIFP-TRC-015 — SHALL**: An external conformance claim SHALL be denied before successful G11 closure or while the effective result of any applicable run-stage requirement is missing, pending, failed, errored, or invalidly not applicable.
- **TIFP-TRC-016 — SHALL**: External-claim requirements SHALL be verified at presentation time against immutable cited hashes and SHALL NOT be counted as missing run-stage trace events.
- **TIFP-TRC-017 — SHALL**: The effective result for a retried requirement or gate SHALL be the highest-sequence valid trace event that explicitly references the superseded event and immutable remediation-resolution evidence.
- **TIFP-TRC-018 — SHALL**: A later passing retry event SHALL resolve but SHALL NOT delete, overwrite, or conceal the superseded failure or error event.
- **TIFP-TRC-019 — SHALL**: A retry event lacking a valid `supersedes_event_sha256` or `resolution_record_sha256` SHALL NOT replace the prior effective failure or error.
