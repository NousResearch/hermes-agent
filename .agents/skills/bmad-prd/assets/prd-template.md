# PRD Template

## Essential Spine *(almost always present)*

```markdown
---
title: {Product Name}
created: {YYYY-MM-DD}
updated: {YYYY-MM-DD}
---

# PRD: {Product Name}
*Working title — confirm.*

## 0. Document Purpose
[1 paragraph: who this PRD is for (PM, stakeholders, downstream workflow owners), how it's structured (Glossary-anchored vocabulary, features grouped with FRs nested, assumptions tagged inline and indexed). If UX work or other inputs already exist, name them here and reference where they live — this PRD builds on them, it does not duplicate.]

## 1. Vision
[2-3 paragraphs: what this is, what it does for the user, why it matters. Compelling enough to stand alone.]

## 2. Target User

### 2.1 Jobs To Be Done
[Bulleted. Emotional, social, functional, contextual — whichever apply. Even "this is for me as the builder" is a valid framing for a hobby project.]

### 2.2 Non-Users (v1) *(add when the audience boundary is non-obvious)*
[Who this is explicitly not for in v1.]

### 2.3 Key User Journeys
*Named-persona narratives the product enables. Numbered globally as UJ-1 through UJ-N. FRs reference journeys by ID inline ("realizes UJ-3"); SMs may also cross-reference. If a UX doc already exists, mirror its UJ IDs here and point to the source.*

**Default shape:** a named scene with entry state, path, climax, and resolution. Each beat forces specificity the team would otherwise leave implicit — auth assumptions, screen order, what tells the user value landed. Read together as a short narrative; the example below shows the form.

- **UJ-1. {One-line title — persona doing the thing.}**
  - **Persona + context:** one line, grounded enough to explain the *why*.
  - **Entry state:** authenticated? which surface? coming from where?
  - **Path:** 3-5 concrete beats — taps, screens, decisions.
  - **Climax:** the moment value is delivered and how the user knows.
  - **Resolution:** state they're left in, what's next.
  - **Edge case** *(optional)*: one real failure mode and what the user does next.

  *Written out, that becomes:*
  > **UJ-3. Priya checks the trip damage before she's even home.**
  > Priya, budgeting on a single income with a new baby, finishes a grocery run and gets in the car. Already authenticated via biometric on a previous session. She opens the app, taps the FAB camera, and scans the receipt. The app OCRs the total and shows a single-screen overlay: this trip $84.20, weekly cap $250, $172.10 remaining, three days left in the week. She closes the app and drives home. **Edge case:** if she scanned a receipt earlier today, the app asks whether this replaces or adds to that trip before counting it against the cap.

- **UJ-2. ...**

**Scope dial:**
- **Lighter** — hobby/solo, library/CLI, or when the UJ is essentially a JTBD restated: a single sentence works (`{Persona}, {context}, {what they do and why}.`).
- **Heavier** — auth, multi-device handoff, complex navigation, or anything feeding downstream UX/architecture: add a numbered Flow, an Edge cases list, and a capability → FR mapping (`The system must {capability}. → FR-N`).

## 3. Glossary
*Downstream workflows and readers must use these terms exactly. FRs, UJs, and SMs use Glossary terms verbatim; introducing a synonym anywhere in the PRD is a discipline violation. If §4 introduces a new domain noun, add it to the Glossary in the same pass.*

- **Term** — Definition. Relationships to other Glossary terms. Cardinality where relevant.
- **Term** — ...

[Every domain noun the rest of the document uses. Defined once. No synonyms anywhere else in the PRD.]

## 4. Features
*Each subsection is a coherent feature: behavioral description first, FRs nested under it, optional feature-specific NFRs and notes. FRs are numbered globally (FR-1 through FR-N) so downstream artifacts have stable references even if features get reorganized. Reference user journeys by ID inline ("realizes UJ-2") where the chain matters.*

### 4.1 {Feature Name}
**Description:** [Behavioral narrative — how this feature works, who uses it, the user experience, edge cases. Realizes UJ-X, UJ-Y. Use Glossary terms exactly. Embed inline `[ASSUMPTION: ...]` tags where you inferred without confirmation.]

**Functional Requirements:**

#### FR-1: {Short capability name}

[Actor] can [capability] [under conditions]. Realizes UJ-X.

**Consequences (testable):**
- {Specific testable condition, e.g. "System returns HTTP 429 when request rate exceeds 100/sec per merchant."}
- {Another testable condition.}

**Out of Scope:** *(optional — what this FR explicitly does NOT cover)*
- {bound}

#### FR-2: ...

**Feature-specific NFRs:** *(only if any apply uniquely to this feature)*
- Performance / security / accessibility / etc. specific to this feature.

**Notes:** *(optional — open questions specific to this feature, `[NOTE FOR PM]` callouts)*

### 4.2 {Feature Name}
...

## 5. Non-Goals (Explicit)
[Bulleted. What this product is *not* and what it will *not* do in v1. Does outsized work for downstream readers and workflows — prevents the "let me also add this nearby thing" failure mode at every level (epic, ticket, code). Inline `[NON-GOAL for MVP]` callouts within §4 Features cover deferred items within features; this section captures the broader "we are not building X / we are not becoming Y" statements.]

## 6. MVP Scope

### 6.1 In Scope
[Bulleted, crisp.]

### 6.2 Out of Scope for MVP
[Bulleted. Each item with a one-line reason if the reason matters. Mark items deferred to v2/v3 explicitly. Add `[NOTE FOR PM]` callouts where a deferred item is emotionally load-bearing — flags it for revisit if timeline permits.]

## 7. Success Metrics

*Each SM cross-references the FR(s) it validates. Counter-metrics counterbalance specific primary or secondary metrics.*

**Primary**
- **SM-1**: Metric — definition, target. Validates FR-X, FR-Y.

**Secondary**
- **SM-2**: Metric — definition, target. Validates FR-Z.

**Counter-metrics (do not optimize)**
- **SM-C1**: Metric — why this should *not* be optimized. Counterbalances SM-1.

[Length scales with stakes. Hobby/utility PRD: a single sentence may be enough ("Success: I use this weekly and don't abandon it after a month"). Public launch / enterprise: full quantitative breakdown with measurement methods. Counter-metrics are as load-bearing as primary metrics — they prevent the architect from optimizing the wrong thing and the dev from gaming the wrong target.]

## 8. Open Questions
[Numbered. Things still unknown — they become future tickets or follow-up research, not silent gaps.]

## 9. Assumptions Index
*Every `[ASSUMPTION]` from the document, surfaced for explicit confirmation:*
- Inline assumption from §X.Y — short description.
- ...
```

---

## Adapt-In Menu *(add the clusters the product calls for)*

### Cross-cutting quality and shape *(most non-trivial PRDs)*
- **Cross-Cutting NFRs** — system-wide non-functional requirements not tied to a single feature (performance, security, reliability, observability). Add when system-wide quality attributes are meaningful.
- **Constraints and Guardrails** — Safety, Privacy, Cost. Subsection per cluster. Add when any of these are real concerns.
- **Why Now** — add when timing is load-bearing (a market shift, a technology enabler, a regulatory deadline). Drop when timing is incidental.

### Consumer / branded products
- **Aesthetic and Tone** — visual references, anti-references, voice/tone for any product-generated text.
- **Information Architecture** — top-level surfaces, navigation, screens.
- **Monetization** — free vs. paid, pricing assumptions, ads policy.
- **Platform** — web, mobile, PWA, native, v1 vs. v2+.

### Enterprise initiatives
- **Stakeholders and Approvals** — who must sign off, at what stage.
- **Risk and Mitigations** — operational, security, business, reputational risk register.
- **ROI / Business Case** — quantified benefit, cost, payback period.
- **Operational Requirements** — SLAs, RTO/RPO, support tier, on-call expectations.
- **Integration and Dependencies** — SSO, existing enterprise systems, data sources, downstream consumers.
- **Rollout and Change Management** — phased rollout plan, training, internal communication.
- **Data Governance** — residency, sovereignty, classification, retention.
- **Audit Trail / Decision Provenance** — formal documentation requirements for regulated environments.

### Regulated domains
- **Compliance and Regulatory** — HIPAA, PCI-DSS, GDPR, SOX, SOC 2, Section 508 / WCAG 2.1 AA, FedRAMP, etc. — whichever apply. If any item needs depth, add a `[NOTE FOR PM]` callout to revisit or move to an addendum.

### Developer products (libraries, APIs, CLIs, SDKs)
- **API Contracts / Public Surface** — endpoint shapes, breaking change policy.
- **Versioning and Deprecation Policy**.
- **Performance Budgets** — latency, throughput, resource use.
- **Language / Runtime Targets and Dependency Policy**.

### Embedded / hardware
- **Hardware Constraints** — memory, power, form factor.
- **Deployment and Update Mechanism** — OTA, manual, image-based.
- **Environmental and Reliability Requirements**.

### Small-scope all-inclusive *(use when scope is 1-2 stories' worth and the user wants a single captured artifact — chosen during the Right-skill check in Discovery)*
- **Stories** — story-level specs listed inline at the end of the doc. Each story: *"As a [persona], I can [action] [under conditions]. Acceptance: [testable criteria]."* Numbered Story-1, Story-2, ... for reference. Pair with very lean §1 Vision, §2 Target User (often just JTBD + one UJ), §3 Glossary (handful of terms), §4 Features (often a single feature), §6 MVP Scope (in/out very tight). The whole doc fits on a page or two and captures intent + implementable stories in one place. If the user doesn't want the captured artifact at all, `bmad-quick-dev` is the better path — this cluster is only for "I want a doc *and* the stories."

