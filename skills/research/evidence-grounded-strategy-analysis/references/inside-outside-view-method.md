# Inside/outside-view method for Elicit

## Structural role

The inside view and outside view belong inside Elicit because a serious plan needs both the case-specific causal story and the historical base rate. The outside view is a reusable method, but its use is a **governing trigger** whenever Elicit produces a forecast, estimate, budget, timeline, adoption expectation, or go/no-go recommendation.

The outside view sets the prior. The inside view may adjust it only with evidence that the focal case differs in ways that predict outcomes.

## Starting without a corpus

Elicit will not initially have a complete reference corpus for every problem. When an outside view is triggered:

1. Define the forecast target and time horizon.
2. Predeclare the causal reference class and selection rules.
3. Use active intelligence to retrieve comparable successes, failures, delays, overruns, cancellations, and incomplete cases from sourced public documents, datasets, and authorized internal evidence.
4. Preserve provenance, extraction criteria, missingness, and outcome definitions.
5. Save verified cases into the governed internal corpus so later runs can reuse and expand the class.

Search must be anchored to the forecast target and causal process, not to cases that support the inside-view story.

## Outside-view map

Use this compact output:

> **Target → Reference class → Selection rules → Cases and provenance → Base-rate distribution → Evidence-backed adjustments → Independent inside forecast → Blended forecast → Confidence → Decision threshold → Actual result**

### 1. Target
Specify the measurable outcome, horizon, and success or failure threshold.

### 2. Reference class
Choose cases produced by substantially the same causal process and constraints. Distinguish structural comparability from topical or visual resemblance.

### 3. Selection rules
Freeze inclusion and exclusion rules before inspecting outcomes. Include failures, abandonment, and missing cases where possible.

### 4. Distribution
Report sample size, missingness, success/failure rate, median, relevant percentiles, dispersion, overruns, delays, and tail risk—not merely an average.

### 5. Independent inside forecast
Build the focal-case forecast separately when practical so the project narrative cannot quietly reshape the reference class.

### 6. Adjustments
List material differences from the class. An adjustment is allowed only when evidence shows that the factor predicts outcomes across cases. Predeclare direction and cap size.

### 7. Blend
Anchor on the outside-view distribution. Move toward the inside forecast only in proportion to reliable case-specific evidence; do not default to a 50/50 average.

### 8. Decision and learning
State uncertainty, the decision threshold, and what action follows at each threshold. Preserve the forecast before the outcome, compare actual results against both views, and add the case back to the corpus.

## Guardrails

- **Reference-class shopping:** Test at least one broader and one narrower defensible class and report whether the decision changes.
- **Survivorship bias:** Include failed, delayed, cancelled, and incomplete cases.
- **Surface comparables:** Match causal process, incentives, maturity, constraints, and selection mechanism—not labels alone.
- **Double counting:** Assign each fact once: class definition, adjustment, or inside model.
- **Narrative override:** The burden of proof rises with the size of departure from the base rate.
- **False precision:** Show sample size, missingness, dispersion, and confidence.
- **Unfalsifiable forecasts:** Freeze assumptions, forecast, confidence, and threshold before observing the result.

## Three-sentence explanation

The inside and outside views belong in Elicit because every serious plan needs both the case-specific story and the historical base rate: the outside view sets the prior, while the inside view can adjust it only with evidence that this case is genuinely different. Elicit produces an independent forecast from each, reconciles them into a documented range and decision threshold, and later compares both with the actual result. When no corpus exists, Elicit defines the reference class first, then uses active intelligence to retrieve comparable successes and failures from sourced public documents, datasets, and its governed internal corpus while preserving provenance and selection rules.
