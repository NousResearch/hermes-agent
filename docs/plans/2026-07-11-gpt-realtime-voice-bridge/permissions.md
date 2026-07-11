# Permissions

## Allowed Scope

### Run 1 — backend security/session contract
- `hermes_cli/web_server.py`의 기존 authenticated Desktop API router
- `tests/hermes_cli/test_web_server.py`
- 필요 시 신규 Realtime session helper 1개와 대응 focused test 1개 (`hermes_cli/`, `tests/hermes_cli/`)
- Desktop backend config/type schema의 Realtime voice 필드

### Run 2 — renderer transport/state
- `apps/desktop/src/hermes.ts`
- `apps/desktop/src/app/chat/composer/hooks/use-voice-conversation.ts`
- 신규 `apps/desktop/src/lib/realtime-voice-session.ts` 및 focused tests
- 관련 voice state/type 파일

### Run 3 — UI/fallback/E2E
- 기존 Desktop voice controls/settings의 최소 변경
- voice focused tests 및 E2E fixture
- 사용자 문서의 Realtime voice 설정 절

Run 1 preflight에서 실제 route owner가 `hermes_cli/web_server.py`임을 확인했다. `apps/desktop/backend/**` 가정은 폐기한다.

## Do Not

- 기존 Agent/tool execution loop, gateway authorization, MCP registry를 우회하거나 복제하지 않는다.
- API key를 renderer/localStorage/log/session transcript에 저장하지 않는다.
- 관련 없는 refactor, Google Meet plugin 변경, CLI voice 변경을 하지 않는다.
- commit/push, 서비스 restart, live credential 사용은 별도 승인/사용자 입력 없이 하지 않는다.
- 기존 STT/TTS 기본 동작을 변경하거나 자동 fallback이 실패를 숨기게 하지 않는다.

## Sensitive Paths

- `.env`, auth/token files, provider credentials: read/write 금지.
- gateway authorization/session persistence 변경: scope escalation.

## Scope Escalation Rule

public API, auth, credential persistence, Hermes tool policy 또는 12파일 초과가 필요하면 Worker는 즉시 `NEEDS_SCOPE_APPROVAL`로 중단한다.

## Worker Output Format

각 run은 `runs/<run-id>/`에 `worker_report.md`, `changed_files.txt`, `commands.log`, `test-results.log`, `diff.patch`, `blockers.md`를 기록한다. `USED_ADRS` 또는 `GRAPH_RATIONALE`, `REUSABLE_FOR_SKILL` 구조화 필드를 포함한다.

## Rollback Plan

- Realtime feature flag/config를 off로 두면 기존 voice mode만 사용한다.
- 각 run을 독립 commit 가능 단위로 유지한다.
- 장애 시 renderer Realtime 진입점만 비활성화하고 기존 `/api/audio/transcribe`, `/api/audio/speak`를 유지한다.

## Failure Recovery Runbook

1. WebRTC 실패: connection state와 OpenAI error event를 redacted 기록하고 UI에서 legacy mode 선택을 제공한다.
2. secret 발급 실패: retry 폭주 없이 1회 사용자 재시도.
3. Hermes turn 실패: Realtime response를 중단하고 기존 gateway 오류를 표시한다.
4. 중복 transcript: client turn id/idempotency key로 차단하고 테스트 실패 처리한다.
5. disconnect: media tracks, peer connection, data channel, timers를 모두 정리한다.
