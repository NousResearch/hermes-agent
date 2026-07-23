# Wave 1 — memory proposal freshness

## Digest

- `stage_write()` records a payload but no target revision.  Approval claims
  only serialize queue ownership; `apply_memory_pending()` replays directly
  into the live `MemoryStore`.
- `MemoryStore` already owns target-scoped sidecar locks.  The safe minimal
  V2 is a raw-byte SHA-256 snapshot at staging and the same digest comparison
  inside that very lock immediately before a single or batch mutation.
- A V1 or malformed proposal must fail closed and must never be upgraded from
  the current target state during approval.  The existing receipt ledger is
  intentionally not a memory-mutation bridge.
- `agent/learning_mutations.py` has a direct memory writer outside the shared
  lock; it must participate in that lock for Hermes-internal serialization.

## Evidence

`tools/write_approval.py:179-245`, `tools/memory_tool.py:253-612,833-944,
1062-1080`, `hermes_cli/write_approval_commands.py:108-145`, and
`agent/learning_mutations.py:65-80,144-197`.

## EXPAND

- DEAD END: outcome receipt에서 memory proposal으로 이어지는 기존 bridge — 설계·코드·참조를 모두 확인했으나 현 V1에는 의도적으로 없음.
- DEAD END: `apply_memory_pending`의 추가 production caller — CLI shared approval handler 외에는 테스트만 확인됨.
- LEAD: stale/legacy proposal을 자동 `expired` 처리하는 lifecycle — WHY: 안전성은 유지하면서 영구 재시도 대기열을 없앨 수 있음 — ANGLE: `handle_pending_subcommand`에 terminal outcome을 추가해 approve/reject 출력과 claim 완료 정책을 검토.
- LEAD: lock 비협력 외부 writer 대응 수준 — WHY: file lock만으로 모든 OS process를 직렬화할 수 없음 — ANGLE: 플랫폼별 advisory lock 보장과 atomic replace 경쟁 정책을 별도 threat-model로 명시.
