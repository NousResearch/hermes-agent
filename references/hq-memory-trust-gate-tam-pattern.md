# HQ Memory Trust Gate TAM Pattern

## Overview

TAM Memory (`yishutian37-commits/trustworthy-agent-memory`)는 Python 표준 라이브러리 기반으로 write gate + faithfulness gate + negative memory index를 제공하는 최소 참조 구현이다. 본 문서는 이 패턴을 HQ memory architecture에 매핑한다.

## TAM 패턴 핵심

- write gate: 메모리 적재 전 정책 팩터(policy_factor)와 신뢰도 스코어(trust_score)로 allow/review/block 3way 필터링.
- faithfulness gate: 모델이 draft를 생성한 후 메모리 제약 위반 여부를 사후 검증.
- negative memory index: 신뢰할 수 없는 사실에 대한 접근을 차단하거나 경고.

## HQ 매핑

| TAM 레이어 | HQ 대상 | 비고 |
|---|---|---|
| write gate | `hq_memory_learning_gate.py` / `hq_control_eval.py` | ingest 파이프라인 앞에 deterministic 정책 검증 체인 추가 |
| faithfulness gate | `hq_harness_validator.py` recall 후 | draft 응답의 메모리 일관성 사후 검증으로 allow/review/block |
| negative memory index | `learning-notes` promotion gate | 유해/오래된 Lessons를 자동 경고 |

## 안전 경계

- write gate는 deny-by-default. block 시 대체 메시지 제공.
- faithfulness gate는 허용/검토/차단의 3way 리턴; 대체 응답은 메모리와 별도 감사 로그 필요.
- 메모리 삭제/merge는 감사 로그를 거치도록 하고, policy_version + audit_id를 기록.

## Prompt-injection 방어 패턴 (arXiv 2604.18248)

TAM write gate + faithfulness gate만으로는 writer-side injection, retriever-side poisoning, faithfulness-gate bypass를 막을 수 없다. 다음 7가지 기법을 memory boundary 레이어에 매핑한다:

| detection technique | memory boundary 적용 위치 | HQ 패턴 |
|---|---|---|
| forensic linguistics | remember/recall 경계 입력 샘플링 | `hq_memory_learning_gate.py` ingest 필터 앞에 의심 문장 샘플러 추가 |
| stylometric fatigue | 재사료된 메모리 항목 감지 | negative memory index와 결합하여 반복 adversarial 문장 패턴 경고 |
| bioinformatics local alignment | retrieval 결과와 인접 문맥 유사도 | recall 결과 1-hop 컨텍스트와 alignment 점수 산출, 임계값 초과시 `review` |
| cross-domain anomaly score | 다중 소스 memory hit 이상 감지 | memorywire 4type + external store hit 분포가 갑자기 바뀌면 block |
| adversarial durability test | gate 회피 시도 탐지 | recall→draft 파이프라인에 deterministic 동치 검사 추가 |
| ensemble detector | 단일 기법 우회 경계 | allow/review/block 결정을 3개 이상 기법의 다수결로 전환 |
| calibrated abstention | 불확실 구간 차단 | faithfulness gate return에 `abstain`을 추가하여 불확실한 메모리 반입 차단 |

구현 순서:
1. write gate 앞에 forensic linguistics 샘플러를 Python-stdlib-only로 추가.
2. recall 후 faithfulness gate 앞에 cross-domain anomaly score 계산.
3. TAM negative memory index에 stylometric fatigue 패턴 추가.
4. allow/review/block → allow/review/block/abstain으로 확장.

## 참고 경로

- GitHub: `yishutian37-commits/trustworthy-agent-memory`
- arXiv 2606.01138 `memorywire`
- arXiv 2505.11942 `LifelongAgentBench`
- arXiv 2604.18248 `Beyond Pattern Matching: Prompt Injection Detection`
- Planner source: `cron_331ecf7312ab_20260623_010635` assistant message (## Response 섹션)
- HQ 참조: `references/hq-memory-learning-gate-and-benchmark-eval.md`
