# HQ LedgerAgent Execution-State Mapping

## Overview

LedgerAgent (arXiv `2606.20529`, 2026-06-18)는 tool-calling agent에 대해 별도의 ledger에 task state를 유지하고 environment-changing tool call 전에 policy constraint를 사전 차단한다. 본 문서는 이 패턴을 HQ execution-state capsule + atomic-fact memory와 결합한 매핑이다.

## LedgerAgent 핵심

- 별도 ledger 레이어가 task state를 유지.
- policy constraint를 graph node에 부착.
- restore 시 policy re-check.
- append-only 원칙 준수.

## HQ 매핑

| LedgerAgent | HQ 대상 | 비고 |
|---|---|---|
| policy-attached graph node | execution-state capsule + atomic-fact memory | policy constraint를 graph node에 부착한 구조를 HQ capsule schema에 반영 |
| pre-call policy re-check | `hq_harness_validator.py` pre-dispatch | environment-changing tool call 전 HQ pre-dispatch 정책 재검증 |
| append-only ledger | atomic-fact memory | append-only + signed provenance 원칙을 HQ memory write path에 일부 적용 |
| restore-time re-check | session restore / cron tick | session/cron tick 시작 시 execution-state capsule에서 policy re-check |

## 안전 경계

- ledger graph는 append-only로 유지.
- policy constraint 변경은 별도 감사 로그 필요.
- future: MCP contract extension으로 signed provenance 연동.

## 참고 경로

- arXiv: `2606.20529` LedgerAgent
- Planner source: `cron_331ecf7312ab_20260622_010031` assistant message `35433`
- HQ 참조: `references/hq-harness-validator-eval-runner-v0.md`
