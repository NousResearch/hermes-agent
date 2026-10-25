---
name: evidence-grounded-strategy-analysis
description: "Use when turning transcripts into grounded strategy."
version: 1.3.1
author: Shaun Overton and Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [qualitative-analysis, strategy, transcripts, provenance, adversarial-review]
    related_skills: [research-paper-writing]
---

# Evidence-Grounded Strategy Analysis

Use this skill when asked to infer an organization's strategy, operating model, software opportunity, or implementation sequence from interviews, meeting transcripts, internal documents, and a constrained research corpus.

## When to Use

- Discovery interviews or meeting transcripts must become decision-ready findings.
- Internal evidence and external corpus support must remain distinguishable.
- A client-facing strategy report needs exact citations plus a separate restricted evidence dossier.
- Recommendations involve material uncertainty, competing end states, or adoption risk.

Do not use this skill for simple transcript summaries, transcription, or generic web research without an organizational decision to support.

## Core rule

Keep two dimensions separate internally:

1. **Evidence class** — `stated`, `inferred`, or `unknown`.
2. **Verdict** — `supported`, `qualified`, `disputed`, or `no corpus citation`.

A claim can be directly stated yet disputed by another account. An inference can be well supported. An internal fact can legitimately have no external corpus citation. Never collapse these dimensions into one confidence label.

### Client-facing findings are not the evidence ledger

For an initial-findings report, preserve classifications and verdicts in a separate Evidence and Verification Dossier rather than displaying grades, corpus identifiers, or provenance scaffolding throughout the argument. The client report should communicate the point directly and use footnotes or endnotes to exact quotations, timestamps, or source locations. Quote directly in the finding only when the original language adds genuine weight or clarity. Put methods, limitations, source documentation, and unresolved questions in dedicated sections or appendices.

### Engagement isolation happens before ingestion and context construction

Never place a prior client's report, transcript, identity, quotations, facts, metadata, attachments, links, or embedded objects anywhere beneath the current engagement root or in analysis, drafting, validation, or review context. Prompt-level instructions such as “do not mention the prior client” are not a privacy control. Require human provenance attestation and an upstream source-of-origin check before ingestion; ambiguous provenance fails closed. For extraction through drafting, permit only the hash-pinned client-neutral protocol plus manifest-listed current-engagement evidence and instructions. Validation/review may additionally receive this run's hash-bound generated artifacts through an explicit review-only data class; never reclassify report or dossier text as instructions. Disable ambient conversation history, persistent memory, prior run output, template bodies, and unlisted retrieval/tool context. See `references/transcript-to-findings-delivery.md` for the practical two-artifact workflow. For the complete released Transcript-to-Initial-Findings Protocol v1.0, load `references/tifp-v1.0-part-1.md` and `references/tifp-v1.0-part-2.md` in that order; the byte concatenation has SHA-256 `9b56f44d11a8b9c599b4a9968dbc0a16604592f1f55ccd87c69e68f6c4acadaf`.

### Conformance requires an instrumented runner

Do not claim TIFP conformance from prompt isolation, generated reports, hashes, or a generic delegated review alone. Before **every** model invocation, the execution layer must already have frozen the protocol-required stage-context, outbound-context, and model-service-boundary records and must be capable of preserving the exact raw response plus response-capture records afterward. These controls cannot be reconstructed retroactively from a subagent summary or final files. If the active runtime cannot expose and bind the exact outbound payload and returned response bytes, use this skill as a rigorous analysis/delivery guide, label the run nonconforming, and do not release it as a TIFP-validated package.

## Workflow

1. **Establish source boundaries**
   - Inspect any user-supplied link or primary source directly before relying on summaries, prior-session context, or search results.
   - Record the source files, duration/page scope, corpus boundary, and provenance limitations.
   - Before launching a frozen-evidence run, verify every input path exists, recompute the expected digest, and confirm reconstructed segment count/bytes; do not trust a prior-session note that an extraction artifact still exists.
   - If new evidence arrives after the run freezes its snapshot, either rerun the panel or add it only as a separately attributed late input in the final deliverable. Never imply that the panel analyzed evidence outside its manifest.
   - If speaker diarization is unreliable, attribute quotations to the meeting or transcript rather than inventing named speakers.
   - Label manual simulations plainly; do not imply certification by an unavailable agent/checker pipeline.
   - For Plaud public-share recordings, use `references/plaud-shared-source-ingestion.md` to preserve the complete timestamped transcript and evidence hashes without treating generated highlights as primary evidence.

2. **Verify model/backend independence**
   - An analysis framework such as Elicit must use the intelligence configured for the invoking agent or an explicitly selected interchangeable backend; it must not silently require one vendor or CLI.
   - Before requesting provider authentication, trace the model invocation path. A hard-coded provider dependency is an implementation defect, not a user setup requirement.
   - Preserve the same structured-output schema, corpus-tool boundary, provenance, cost/timing records, verifier behavior, and fail-closed semantics across backends.
   - Keep provider-specific adapters optional. Never force unrelated OAuth merely because one adapter is the current default, and never substitute fabricated output when no backend is available.
   - For Hermes-hosted Elicit adapters, follow `references/provider-neutral-elicit-runtime.md` for active-session inheritance, timeout cleanup, invocation metadata, and truthful budget semantics.

3. **Extract before synthesizing**
   - Read the complete source, not keyword excerpts alone.
   - Build an evidence matrix covering desired future state, customer promise, economics, operating model, constraints, contradictions, software hypotheses, adoption capacity, and missing decisions.
   - Preserve exact quotations and exact source segment headers.
   - For outreach or warm introductions, isolate the target person’s aspiration, practical friction, and self-authored closing takeaway. Confirm identity through an explicit named handoff plus matching business details when diarization is absent; do not attribute adjacent participants’ examples to the target.
   - Frame the meeting around the target’s desired outcome (clarity, scalability, market position, or operating leverage), not around “AI” by default. Verify who was actually offered a follow-up and never transfer an invitation from one host to another.

4. **Form competing strategic hypotheses**
   - Preserve materially different end states instead of blending them into false consensus.
   - Separate customer-facing strategy from internal operating-model design.
   - Treat proposed technology architectures as hypotheses until comparative evidence exists.

5. **Require inside/outside-view forecasting when decisions depend on estimates**
   - Trigger an outside-view layer whenever the task includes a forecast, budget, timeline, adoption estimate, success probability, or go/no-go threshold. Implement it as a reusable Elicit method, but treat the trigger as a governing requirement rather than one optional perspective among the selected methods.
   - Define the forecast target and a causal reference class before examining outcomes. Freeze inclusion/exclusion rules and include failed, delayed, abandoned, and incomplete cases where possible.
   - When no governed corpus yet exists for the class, use active intelligence to retrieve cases from sourced public documents, datasets, and authorized internal evidence. Anchor retrieval to the forecast target and causal process—not to examples that merely resemble or support the inside-view narrative. Preserve provenance so verified cases can accumulate into the governed internal corpus.
   - Produce the inside-view forecast and outside-view distribution independently when practical. The outside view establishes the prior; case-specific evidence may adjust it only when the difference is outcome-predictive, quantified, and not already embedded in reference-class selection.
   - Test at least one broader and one narrower defensible reference class, report whether the decision changes, and prohibit double-counting the same favorable fact in both class selection and adjustment.
   - Record the blended forecast, uncertainty, decision threshold, assumptions, and realized result. Add the completed case back to the corpus so future base rates improve.
   - Use `references/inside-outside-view-method.md` for the compact mapping template and guardrails.

6. **Retrieve corpus evidence narrowly**
   - Search the fixed corpus for principles that genuinely bear on the decision.
   - Retrieve exact passages and source metadata, not search snippets alone.
   - Use external research to support methods or constraints; do not make it certify organization-specific facts it cannot know.
   - Abstain when the corpus is off-topic or too thin.

7. **Build the dossier before the client narrative**
   - In the restricted Evidence and Verification Dossier, give each material finding a fixed anatomy:
     - Observation.
     - Verdict.
     - Evidence class.
     - Exact internal evidence.
     - Exact corpus support, if relevant.
     - Qualification or unresolved decision.
   - Then translate the supported argument into client-facing prose. Keep grades, verdict labels, corpus IDs, and provenance mechanics out of the body; use stable footnotes/endnotes to dossier evidence IDs.
   - Use a direct quotation in the finding only when its wording adds genuine explanatory or persuasive weight.

8. **Derive recommendations explicitly**
   - Mark proposed flywheels, pilots, timelines, metrics, and architectures as inferences or planning assumptions.
   - Prefer risk-controlled, reversible tests when the evidence does not justify migration or replacement.
   - State what result would falsify the recommendation.

9. **Adversarial review**
   - Dispatch an independent reviewer against both report and primary sources.
   - Ask it to check every finding for quote accuracy, timestamp support, overclaiming, internal contradiction, verdict choice, citation stretch, and **speech-act accuracy**: distinguish an adopted position from a hypothetical, proposed wording, interpretation of an absent person, question, or rejected option.
   - Require the reviewer to identify counterevidence that narrows broad statements such as “no data is trusted,” “the company has decided,” or “change capacity is binding.” Scope conclusions to the participants and evidence actually represented.
   - A review must inspect the current artifact. If the report changes after dispatch, supersede the prior review or perform a new final-artifact review.
   - Track every outstanding review batch. Do not circulate or finalize while any reviewer is still running; an early checkpoint is not a reviewed deliverable.
   - Reconcile each blocker explicitly: edit, reject with evidence, or disclose as unresolved. Then rerun mechanical verification on the edited artifact.
   - Continue the edit → fresh-review loop until the current artifact passes or remaining blockers are disclosed. A later review may surface a different omission once an earlier blocker is fixed; an earlier partial pass does not cover the edited document.
   - If a framework verifier fails because claims are unaccounted for or acceptance criteria lack traceable test records, preserve that disposition. The deliberation may still be used as a challenge record, but not as certification; present affected requirements as proposals, questions, or tests rather than settled facts.

10. **Mechanical and semantic verification**
   - Confirm every cited timestamp range exactly matches a source segment header.
   - When evidence spans consecutive segments, cite each segment separately; never invent a combined range.
   - Exact timestamp containment is necessary but insufficient: inspect quotation context to ensure proposed language is not presented as an adopted strategy and an absent leader’s attributed vision is not treated as direct testimony.
   - Verify every corpus passage identifier resolves through the corpus interface.
   - Cross-check the “citations used” inventory against citations actually applied in the report body; remove unused entries or apply them explicitly.
   - Check that every material finding has an evidence class and every recommendation discloses assumptions.
   - For pilots, state what they cannot establish (for example enterprise-wide truth, final architecture selection, organization-wide adoption capacity, or strategic purpose).
   - Record an artifact hash when useful for binding the review to the delivered file.

## Output shape

### Client-facing Initial Findings Report

1. Bottom-line strategic answer.
2. Practical findings in natural professional prose.
3. Competing end states or contradictions where decision-relevant.
4. Testable operating/flywheel hypothesis.
5. Reversible first intervention.
6. Unknowns that would change the recommendation.
7. Concise methods and limitations section.
8. Footnotes/endnotes to exact transcript quotations, timestamps, source locations, or stable dossier evidence IDs.

Do not display evidence grades, verdict badges, corpus passage identifiers, raw paths, model/provider details, or verifier mechanics in the middle of the argument.

### Restricted Evidence and Verification Dossier

1. Source, stage-context, and outbound-context manifests.
2. Evidence ledger with class and verdict.
3. Exact quote/timestamp ledger.
4. Claim-to-source and claim-to-footnote map.
5. Counterevidence, contradictions, assumptions, and unresolved questions.
6. Corpus citations used and important abstentions.
7. Validation records and subject-set preparation metadata through the pre-binding gate.

Keep post-binding review dispositions and publication decisions as immutable append-only control records outside the dossier, each bound to the exact subject-set hash. Never mutate the dossier merely to record its own review or publication outcome.

## Pitfalls

- **Cross-engagement context leakage:** Telling a model not to mention another client does not protect that client's information. Do not put prior-client material into the evidence snapshot or any model-visible context.
- **Template laundering:** Calling a prior report “structural” does not make it safe. Convert it into a client-neutral specification outside the engagement, then verify that no identity, fact, quotation, metadata, or conclusion remains before use.
- **Evidence theater:** Repeating grades, verdicts, provenance labels, and corpus identifiers through the client narrative makes the report harder to read without increasing rigor. Keep them in the dossier and expose the traceability through footnotes/endnotes.
- **Stale or self-invalidating review:** A changed dossier, source manifest, protocol version, instruction set, reviewer-control manifest, validation artifact, or report invalidates the prior subject-set review. Freeze reviewer controls before binding; keep the resulting review disposition and publication record outside the dossier as append-only records. Never mutate a reviewed dossier to add its own disposition.
- **Mis-threaded delivery:** Send protocol/report progress and completion notices as standalone messages unless the user explicitly asked within a matching thread. A background completion event is not permission to reply to an unrelated earlier question; verify the reply target before delivery.

- **False consensus:** Several compatible-sounding aspirations are not necessarily one agreed strategy.
- **Speech-act collapse:** Proposed wording, a hypothetical framework, or one participant’s interpretation of an absent leader is not a direct declaration or adopted organizational position.
- **Organization-wide extrapolation:** “The meeting did not establish X” is often defensible; “the company has no X” may not be when key leaders or frontline stakeholders were absent.
- **Counterevidence suppression:** Global statements such as “zero trust” may coexist with trusted current-period domains. Preserve the narrower truth instead of choosing the most dramatic quotation.
- **Tool-first strategy:** A dashboard cannot resolve a decision with no agreed objective function. Strategy clarification and instrumentation may proceed in parallel: data can compare options but cannot choose purpose or risk preference.
- **Custom-software leap:** “Commercial tools fit poorly” does not prove a custom replacement has better risk-adjusted returns.
- **Pilot overreach:** A narrow reconciliation pilot may expose causes and test feasibility; it cannot by itself establish enterprise-wide truth, select a final architecture, prove adoption capacity, or settle strategy.
- **Citation laundering:** A general standard can support a method but cannot prove an organization-specific conclusion. Describe normative guidance as such, not as empirical validation.
- **Citation-inventory drift:** Do not list a passage as “used” unless the body applies it to a claim.
- **Compressed timestamp ranges:** Combining adjacent segment boundaries creates a citation that does not exist in the source.
- **Speaker invention:** Broken diarization does not justify confident attribution.
- **Arbitrary timelines:** Label 30/60/90-day durations as planning assumptions unless evidence establishes them.
- **Reviewer race:** Any post-dispatch edit invalidates the reviewer’s coverage of the final artifact. Late-arriving reviews must be reconciled before circulation, followed by verification of the edited report.

## Supporting references

- Use `references/adversarial-report-checklist.md` for the final evidence and citation audit.
- Use `references/transcript-grounded-outreach.md` to convert a workshop or discovery transcript into a person-specific warm introduction without speaker or invitation misattribution.
