# HUMAN_APPROVAL_GATES

## Purpose
Define editorial and operational approval gates where human judgment is required before data exposure, promotion, or configuration change.

## Gates

| Gate | Owner/Approver | Trigger | Documentation |
|------|----------------|---------|---------------|
| data promotion | Arthur Lyons | any movement from quarantine to validated | Storage manifest |
| configuration change | Arthur Lyons | env/config.yaml non-drift change | change record |
| schema update | Arthur Lyons | DATA_SCHEMA_V1 change request | schema diff |
| backup restore | Arthur Lyons | production restore from backup | recovery manifest |
| restricted data | Arthur Lyons, Naomi | any Restricted intake | access ticket |
| secret disclosure | Arthur Lyons | any Secret criterion Match | evidence deny confirmation |

## Approval format
- Approval id, initiator, stage, artifact hash, action, date/time.
- No implicit approval.

## Blocking behavior
All human gates must block the process step if approval is missing.
