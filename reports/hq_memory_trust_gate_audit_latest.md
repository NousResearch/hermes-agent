# HQ Memory Trust Gate TAM Alignment Audit

Generated: 2026-06-23 01:06 KST
Build tick: cron_a175e5ca4efc (next 01:40)
Planner source: cron_331ecf7312ab_20260623_010635 ## Response

## Inspected scripts (read-only)

- `C:/Users/82109/AppData/Local/hermes/scripts/hq_memory_learning_gate.py`
- `C:/Users/82109/AppData/Local/hermes/scripts/hq_learning_notes.py`

## TAM write-gate / faithfulness-gate / negative-index alignment

### hq_memory_learning_gate.py
- 현재 역할: learning notes JSON → promotion candidate 필터 + 중복 키워드 집계 + 임시 artifact 제거.
- TAM write gate(ingest filter → policy check)와 유사한 필터링이 `TEMPORARY_ARTIFACT_RE` / `NOISE_RE`에 있음.
- 부족: 정책 팩터(policy_factor), 신뢰도 스코어(trust_score), allow/review/block 결정이 없음. negative memory index 없음. faithfulness gate 없음.
- patch point: `build_memory_learning_gate` 함수 앞에 deterministic ingest 필터(수용/검토/차단) 추가; negative index를 keyword 빈도 + 시간 기반 역가중치로 구현 가능.

### hq_learning_notes.py
- 현재 역할: state.db 최근 세션에서 신호(signal) 키워드 포함 assistant 메시지를 추출, sanitize, 노트 작성.
- TAM write gate와 유사하게 `is_durable_learning_sentence`가 노이즈/임시 artifact를 필터링.
- 부족: write 시 faithfulness 검증 없음(메모리 적재 전 constraint 체크). negative memory index 없음. prompt-injection 방어 없음.
- patch point: `sanitize` 뒤에 메모리 삽입 전 constraint gate(policy_factor + trust_score) 추가; 기존 note를 time-decay + forbidden-topic negative index로 재평가.

## 결론

두 스크립트는 TAM의 "필터" 레이어를 부분 구현했으나, write gate의 결정적 정책 검증, faithfulness gate의 사후 검증, negative memory index의 악성/유해 차단은 결여되어 있다. 현재 context hygiene 30/100이므로 live 코드 변경은 보류하고, 위 참조 문서 업데이트를 완료한 상태.

## 다음 Build 승인 요구

- worktree `hq-harness-validator-v0` cleanup + live trust-gate fixture 삽입: clean-worktree 확보 후 승인 필요.
