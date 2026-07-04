# HQ Harness Awesome Patterns Mapping

## Overview

`ai-boost/awesome-harness-engineering` (★1958, 2026-06-22 기준)에서 수집된 핵심 패턴을 HQ Harness Validator, Context Hygiene, Task Score 등 기존 인프라에 매핑한다. 이 문서는 공정(harness) 설계의 4요소(agent loop, tool interface, context management, control mechanisms)를 HQ의 eval·대시보드·컨텍스트 예산 체인과 연결한다.

## 수집 대상 범위

- 2026년 상반기 핵심 엔트리 위주로, HQ 참조문서와 미중복인 항목을 우선 선별한다.
- 중복이거나 이미 반영된 항목은 참고용으로만 기록하고, 새 매핑 테이블에는 포함하지 않는다.

## HQ에 직접 매핑 가능한 패턴

| 엔트리 | HQ 대상 | 매핑 포인트 |
|---|---|---|
| Anthropic — Harness Design for Long-Running Application Development | `hq_harness_validator.py` + `hq_context_hygiene.py` | 장시간 실행되는 agent loop의 리소스 수명·context fragmentation 방지 규칙과 HQ context hygiene score 연동 |
| Microsoft — Azure SRE Agent (35K+ 자율 사고 처리) | `hq_trace_index.py` / `hq_task_score.py` | 대규모 자율 사건 처리에서의 trace 일관성·recovery checkpoint와 HQ trace observability 연계 |
| Life-Harness (18개 모델 backbone에서 전이 가능한 lifecycle-aware runtime) | `hq_harness_validator.py` | 모델 교체 없이 lifecycle-aware runtime 규칙을 harness conformance fixture로 전환 |
| LangChain — Improving Deep Agents with Harness Engineering (Terminal Bench 2.0 30위→top 5) | `hq_ops_dashboard.py` / eval 패키지 | 모델 교체 없는 성능 개선이 harness 수준에서 발생한다는 점을 HQ improvement velocity score 지표로 변환 |
| Harness design for agent eval | `hq_harness_validator.py` | 하네스의 4요소를 HQ validator의 deterministic checklist로 채택 |

## 적용 지표

- Task Score `harness_conformance`: 매 session/cron tick마다 4요소 존재 여부를 기록.
- Context Hygiene score: harness-induced context fragmentation 이벤트를 별도 카운트로 분리.
- Ops Dashboard: harness pattern coverage % (현재 0/15, awesome list 기준 10개 패턴 중 적용된 것) 노출.

## 참고 경로

- GitHub: `ai-boost/awesome-harness-engineering`
- Planner source: `cron_331ecf7312ab_20260622_010031` assistant message `35433`
- HQ 참조: `agi-readiness-roadmap` §5 (agent state core), `references/hq-harness-validator-eval-runner-v0.md`
