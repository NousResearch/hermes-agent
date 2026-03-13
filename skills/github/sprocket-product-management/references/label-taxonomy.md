# Suggested Sprocket Label Taxonomy

Use the repository's real labels where they already exist. If labels are being created or normalized, this is the target conceptual model.

## Type labels

- `type:bug`
- `type:feature`
- `type:task`
- `type:cleanup`
- `type:research`
- `type:epic`

## Status labels

- `status:triage`
- `status:research`
- `status:ready`
- `status:blocked`
- `status:in-progress`
- `status:review`
- `status:done`

## Optional priority labels

- `priority:p0`
- `priority:p1`
- `priority:p2`
- `priority:p3`

## Optional area labels

Examples:
- `area:api`
- `area:web`
- `area:infra`
- `area:harness`
- `area:data`
- `area:ux`

## Notes

- Every issue should have exactly one primary `type:*` label.
- Every active issue should have one clear `status:*` label.
- Priority is optional but useful when the queue grows.
- Area labels are helpful for routing and reporting but should not replace type/status labels.
