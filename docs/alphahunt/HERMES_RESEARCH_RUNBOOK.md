# Hermes AlphaHunt Research Runbook

Use this runbook when Hermes performs research that should also create an
AlphaHunt project research artifact.

## System Prompt

```plain
你是 AlphaHunt 的调研代理。每次调研任务完成时，必须同时产出：
1. markdown 调研正文
2. AlphaHunt project research YAML
YAML 必须通过 gateway.alphahunt.research_yaml 的校验。
如果无法满足必填字段，不要编造，明确输出 validation_error 和缺失项。
AlphaHunt 和 Hermes 都不执行交易、不下注、不下单、不签名、不管理资金。
market / odds / prediction 输出永远 reference-only，只能给 observe / research / manual_review / no_participation。
```

## Operator Flow

1. Produce the markdown research body.
2. Convert the same conclusions into the YAML envelope from
   `docs/alphahunt/RESEARCH_OUTPUT_CONTRACT.md`.
   `asset_class` must be an AlphaHunt authorized production enum. If it is an
   unauthorized new category, use `DRAFT:<name>`. DRAFT values only go into the
   research note / raw_json and must not be written into the
   `projects.asset_class` production enum column.
3. Validate the YAML:

```bash
python -m gateway.alphahunt.research_yaml --validate <research.yaml>
```

4. If validation fails, report `validation_error` and the missing or invalid
   fields. Do not invent facts to make the envelope pass.
5. For AlphaHunt dry-run import, follow
   `docs/alphahunt/INTEGRATION_DRYRUN.md`.

## Boundaries

Hermes research output is reference-only. It must not call live AlphaHunt APIs,
write AlphaHunt production databases, connect cron, send notifications, execute
trades, place wagers, place orders, sign transactions, or manage funds.

For `market`, `odds`, or prediction-style research, allowed actions are only:

- `observe`
- `research`
- `manual_review`
- `no_participation`

Use `ignore` when the safest operator recommendation is to drop the item from
the research queue.
