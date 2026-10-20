# Issue Reporting Procedure

Use when the user reports an issue with a previous routing decision.

## Procedure

1. Acknowledge the feedback without defensiveness
2. Re-check the Sub-skill registry for a better match
3. If a different service is clearly right, load that skill
4. If unclear, broaden the search: use huaweicloud_search_docs
5. Document the mismatch so the registry can be improved

## Common Corrections

| Wrong Routing                        | Likely Correct           |
| ------------------------------------ | ------------------------ |
| ECS for containers                   | CCE                      |
| OBS for block storage                | EVS (via huawei-ecs)     |
| RDS for cache                        | DCS (via huawei-dds-dcs) |
| FunctionGraph for long-running tasks | ECS or CCE               |
