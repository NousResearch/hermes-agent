# Roadmap Adjustment Pattern — Evidence-Based Reordering

Use when a multi-mission roadmap exists and an audit/authority-map mission produces evidence that the next planned mission is not the correct next step.

## Trigger

An audit or analysis mission (READ_ONLY_AUDIT type) produces findings showing that a planned downstream mission:

- Would create an additional authority layer instead of reducing complexity
- Would operate on inputs that are not yet canonical
- Would be blocked by an unresolved architectural dependency
- Would be premature because a prerequisite consolidation is missing

## Method

1. **Do NOT skip the planned mission or proceed anyway.** Adding another layer on top of unresolved debt compounds the problem.

2. **Do NOT implement the prerequisite immediately inside the audit.** The audit is read-only. The prerequisite itself becomes a new CODE_CHANGE mission.

3. **Produce an explicit roadmap update document** (`docs/superpowers/plans/CLOE_ROADMAP_AFTER_<MISSIONS>.md`) that:
   - Cites the specific evidence from the audit
   - Explains why the original ordering was wrong
   - Inserts the prerequisite mission(s) before the originally-planned mission
   - Re-numbers all subsequent missions
   - Documents the architectural principle that justified the reordering

4. **Update memory** with the new ordering so future sessions start from the correct sequence.

## Concrete example (from CLOE M1-M3)

**Original order:**
```
M4: Canonical Response Policy Resolver
M5: Retrieval & Evidence Context Builder
```

**Evidence from M3 (routing authority audit):**
The audit confirmed 3+ parallel classifiers with incompatible taxonomies, no single intent contract. A Response Policy Resolver requires canonical inputs — without classifier consolidation, adding the resolver would create another authority layer instead of reducing complexity.

**Corrected order:**
```
M4: Canonical Intent Classifier Consolidation (NEW — prerequisite)
M5: Canonical Response Policy Resolver (was M4)
M6: Canonical Router Unification (was M5)
M7: Retrieval & Evidence Context Builder (was M5)
M8: Provider-Enriched Answer Composer (was M6)
```

**Architectural principle:**
> Responsibilities must remain strictly separated. Intent classification must never perform routing. Policy resolution must never generate responses. Routers must never classify. Providers must never decide execution permissions. Governance must never classify conversational intent.

## Pitfall: mission-type confusion

When reordering, verify the mission type stays correct:
- If the new prerequisite is an analysis/mapping/inventory → READ_ONLY_AUDIT
- If the new prerequisite is code implementation → CODE_CHANGE
- The newly inserted mission's type determines its gates (see lah-workflow Gate 5)
