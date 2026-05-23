# Research task metaharness for downstream artifacts

This note records a generalized harness pattern for research tasks whose outputs may later be used in another artifact, workflow, decision, or service. The focus is not a domain-specific evidence structure; it is the meta-level control problem: when research is consumed by a real process, the harness must evaluate expected utility, expected loss, probability of occurrence, and downstream propagation before deciding how the result may be used.

## Scope

Use this pattern when a research task produces evidence, claims, summaries, recommendations, rankings, measurements, source collections, report sections, benchmark notes, decision cards, or other intermediate outputs that may later be rendered, published, operationalized, or used to make decisions.

Non-goals:

- Do not treat a research note, HTML page, deck, PDF, dashboard, or chat response as the source of truth.
- Do not let a `pass` label imply unrestricted service use.
- Do not optimize for answer completeness alone when downstream loss can be high.
- Do not assume classification metrics are always the right evaluation frame.

## Core risk model

Practical use means the research output has consequences. The harness should ask:

```text
Expected net value = Σ P(outcome_i | output, use_context) * benefit_i
                   - Σ P(harm_j | output, use_context) * loss_j
                   - review_cost
                   - delay_cost
                   - recall_or_remediation_cost
```

The same research output can be net-positive for one use and net-negative for another. A weakly sourced trend summary may be acceptable for ideation, but unacceptable for an operational decision. A noisy estimate may be useful when the business loss is smooth and bounded, but dangerous when a threshold error triggers irreversible action.

False-accept and false-reject analysis is one useful lens, especially for admit/block gates, but it is not the whole evaluation model. Select the metric and loss function according to the downstream action.

## Metric and loss-function selection

Choose an evaluation family that matches the service use:

- Binary admit/block decisions: false-accept cost, false-reject cost, precision/recall, threshold-specific expected loss.
- Ranked retrieval or recommendation: precision@k, recall@k, nDCG@k, MRR, expected value@k.
- Numeric forecasts or measurements: RMSE, MAE, MAPE, quantile loss, calibration error, prediction interval coverage.
- Risk scoring: expected calibration error, Brier score, AUROC/AUPRC, threshold-specific expected loss.
- Multi-step decisions: policy regret, cost-sensitive utility, decision-curve analysis, scenario simulation.
- Rare catastrophic harms: tail-risk metrics, worst-case scenario review, stress tests, red-team fixtures, human escalation rather than average accuracy.

The harness should record why the chosen metric fits the downstream use. Do not use accuracy, search relevance, or surface plausibility as a universal proxy for value.

## Required classification dimensions

Each admitted research output should carry enough metadata to decide where it may be used:

- Source identity: publisher, institution, title, version, URL/path, retrieval date, hash or retrieval receipt when available.
- Source authority: primary source, official guidance, vendor/source-of-record, recognized standard, industry practice, organization policy, secondary reference, unverified source.
- Output type: fact, estimate, ranking, forecast, interpretation, requirement, recommendation, prohibition, exception, boundary, evidence item, hypothesis, open question.
- Applicability: jurisdiction or market, product/version, site/context, user role, task family, time validity, exclusion conditions.
- Uncertainty: confidence, error bounds, calibration status, sample limits, unresolved assumptions.
- Benefit if right: value domain, expected magnitude, affected users or workflows, time horizon.
- Loss if wrong or missing: negligible, low, medium, high, critical, catastrophic; with harm domains.
- Probability model: observed frequency, estimated likelihood, confidence in likelihood, tail-risk notes.
- Downstream use: search only, ideation, draft, internal review, customer-facing artifact, operational decision, automated action, compliance-grade assertion.
- Recallability: whether dependent artifacts, decisions, or workflows can be found, invalidated, corrected, and re-issued.

## Decision output

Avoid a single `pass/fail`. Emit an allowed-use scope plus the metric/loss basis:

```yaml
allowed_use_scope:
  search_index: true
  ideation: true
  draft_generation: true
  internal_human_review: true
  customer_facing_artifact: false
  operational_decision: false
  automated_action: false
  compliance_grade_assertion: false

metric_basis:
  selected_metric: precision@10
  reason: downstream user reviews a short ranked source list before drafting
  rejected_metrics:
    - name: overall_accuracy
      reason: does not capture top-k review workflow

expected_value_basis:
  primary_benefit: faster source triage
  primary_loss: reviewer accepts unsupported source as authoritative
  probability_notes: rare but high-impact misuse; requires reviewer receipt before customer-facing use

required_review:
  - domain_sme
  - risk_owner

reasons:
  - source authority is sufficient for background but not for binding claims
  - uncertainty remains about applicability to the target context

loss_if_misused:
  severity: high
  harm_domains:
    - customer_misdecision
    - regulatory_noncompliance
    - operational_disruption

recheck_after: 2026-06-30
recall_requirements:
  - track downstream artifacts using this research packet
  - block auto-publication until reviewer receipt exists
```

## Gate sequence

### Gate 0: Downstream-use contract

Freeze the intended use before collecting evidence:

- artifact or decision types that may consume the research;
- audience: private, internal, customer, public;
- actionability: informational, decision support, operational decision, automated action;
- expected benefit and loss categories;
- whether loss is smooth, thresholded, irreversible, or heavy-tailed;
- reversibility and recallability;
- risk tier and required reviewers;
- prohibited uses.

### Gate 1: Metric/loss-function fit

Declare the metric before evaluation. A research task used for top-k source selection should not be evaluated only by global accuracy. A numeric estimate should not be judged only by a binary pass/fail if RMSE, MAE, or interval coverage better captures downstream loss. A rare catastrophic scenario should not pass merely because average error is low.

### Gate 2: Source admission

Admit sources only with identity, reachability/freshness, authority, applicability, and usage constraints recorded. Secondary or unverified sources may support discovery, vocabulary, or hypotheses, but must not silently support high-loss downstream use.

### Gate 3: Output extraction and traceability

Extract structured records rather than prose-only summaries. Each material assertion, estimate, ranking, or recommendation needs direct evidence references and uncertainty notes. A bibliography without output-level mapping is not enough for high-stakes service use.

### Gate 4: Expected benefit/loss classification

Classify what happens if the output is right, wrong, stale, missing, or applied to the wrong context. Higher-loss outputs require stronger evidence, stricter freshness checks, narrower use scope, and human escalation.

### Gate 5: Missing-critical-coverage check

Do not only validate outputs that exist. For each task or artifact type, define minimum coverage templates for required negatives, exceptions, uncertainty bounds, authority boundaries, safety/legal/security requirements, and evidence records. Missing critical coverage should block higher use scopes.

### Gate 6: Conflict and uncertainty preservation

Unresolved conflicts and uncertainty should remain explicit. The harness should not synthesize conflicting sources or noisy estimates into a single confident answer unless the condition split, precedence rule, model assumption, or reviewer decision is recorded.

### Gate 7: Preservation into downstream artifacts

Before a research output feeds a rendered artifact, generated response, or decision process, check that uncertainty was not removed, conditions were not broadened, caveats were not lost, and unsupported extrapolations were not introduced.

### Gate 8: Blast-radius and recall check

For customer-facing, operational, reusable, or automated outputs, require dependency tracking: which artifacts or decisions used which research packet, how they can be invalidated, and how affected users or operators are notified if the packet is later rejected.

## Expected-value review checklist

For each harness release or major research packet, ask:

- What downstream action will this research enable?
- What is the expected benefit if it is right?
- What is the expected loss if it is wrong, stale, missing, overgeneralized, or applied to the wrong context?
- What are the estimated probabilities, and how confident are we in those estimates?
- Is the loss smooth enough for RMSE/MAE-style evaluation, ranked enough for @k metrics, or threshold/catastrophe-heavy enough to require explicit scenario gates?
- Which metric would make the harness look good while hiding the real service loss?
- What is the highest-loss accepted error this packet could allow?
- What missing information would be most damaging if absent?
- Could a downstream generator weaken, generalize, or omit uncertainty and conditions?
- How far would one bad output propagate?
- Can every dependent artifact or decision be recalled or corrected?
- Is conservative blocking justified, or is the harness creating avoidable opportunity loss for low-risk uses?

## Critical review of the pattern

Strengths:

- Optimizes for downstream expected value rather than retrieval accuracy or plausibility alone.
- Separates research admissibility from service-use permission.
- Supports multiple loss models, including classification losses, ranked retrieval metrics, numeric error metrics, calibration, and tail-risk review.
- Handles omissions, staleness, applicability drift, uncertainty loss, and rendering/generation drift as first-class failure modes.
- Makes conservative blocking defensible for high-loss outputs while allowing lighter use for ideation, drafts, and search.

Known limits:

- Structural metadata gates do not prove semantic correctness.
- Probability and loss estimates can be wrong, biased, or politically shaped.
- Coverage templates require maintenance; stale templates create false assurance.
- Human review can become a bottleneck if use scopes are not risk-tiered.
- Recall tracking must be implemented in the downstream artifact or workflow system; a research note alone cannot guarantee revocation.
- Metric choice can be gamed unless reviewers inspect whether it matches real use.

Minimum viable implementation should combine structured records, declared metric/loss basis, deterministic validators, negative fixtures for recurring failure modes, reviewer receipts for high-loss outputs, and downstream dependency tracking before claiming service-grade readiness.
