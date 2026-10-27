# Lane Contracts

The coordinator compares independent lane outputs. Lanes may be produced by code, models, retrieval, human notes, or domain tools, but each lane should be normalized before coordination.

## Lane Record

```json
{
  "lane_type": "evidence",
  "status": "completed",
  "confidence": "medium",
  "findings": [
    {
      "code": "IDENTITY_OR_LINKAGE_UNVERIFIED",
      "severity": "medium",
      "summary": "The record supports the general need but not claimant linkage."
    }
  ],
  "positives": [
    {
      "code": "DATED_PRIMARY_DOCUMENT_PRESENT",
      "summary": "A dated document is present and legible."
    }
  ],
  "decision": {
    "supports_core_claim": "partial",
    "supports_requested_action_size": "unknown",
    "source_independence": "not_applicable"
  },
  "limitations": [
    "Document authenticity was not independently checked."
  ]
}
```

## Required Fields

- `lane_type`: stable lane name, such as `evidence`, `external_context`, `graph_history`, `policy`, or `domain`. This must match the `NAME` in `--lane NAME=file.json`; mismatches are rejected.
- `status`: `completed`, `missing`, `not_applicable`, or `error`. Required lanes satisfy policy only when `completed`; `not_applicable` is preserved as a valid lane status but does not count as completed evidence for a required lane.
- `findings`: array of concern objects.
- `decision`: lane-specific normalized facts used by the coordinator.

## Finding Severity

- `info`: note only.
- `low`: weak concern or limitation.
- `medium`: material uncertainty that should reduce size or confidence.
- `high`: strong concern that should usually reject, monitor, or require a much smaller action.
- `critical`: hard blocker under ordinary policies.

## Recommended Generic Codes

- `UNSUPPORTED_ON_CURRENT_RECORD`
- `CONTRADICTION_IN_RECORD`
- `IDENTITY_OR_LINKAGE_UNVERIFIED`
- `USE_OF_FUNDS_UNVERIFIED`
- `REQUESTED_ACTION_SIZE_UNSUPPORTED`
- `SOURCE_INDEPENDENCE_WEAK`
- `COPIED_TEXT_OR_TITLE_RISK`
- `SYNTHETIC_CONSENSUS_RISK`
- `REPEATED_RECIPIENT_PATTERN`
- `RELATED_RECORD_NETWORK_CONCERN`
- `POST_ACTION_MOVEMENT_UNREVIEWED`
- `POLICY_AUTHORITY_EXCEEDED`
- `DOMAIN_HARD_BLOCKER`

## Decision Fields The Coordinator Understands

The coordinator is intentionally permissive. It uses these fields when present:

- `supports_core_claim`: `yes`, `partial`, `unknown`, `no`, `contradiction`, or `unsupported`.
- `supports_requested_action_size`: `yes`, `partial`, `unknown`, `no`, or `unsupported`. `no` and `unsupported` reject the current requested action size; `partial` and `unknown` reduce eligibility to a smaller test action under ordinary policy.
- `source_independence`: `strong`, `some`, `weak`, `none`, or `not_applicable`.
- `public_context_plausibility`: `all_found`, `some_found`, `weak`, `not_found`, or `not_applicable`.
- `claimant_linkage`: `verified`, `partial`, `unknown`, `unverified`, or `contradicted`.
- `repeat_or_top_up_support`: `fresh_reason_present`, `weak`, `none`, or `not_applicable`.

Unknown top-level fields are preserved in the coordinator's `lane_details` output but do not affect deterministic rules.
