# Repository Assessment

## 조사 기준

2026-07-24의 `feature/hegi-v2` 브랜치와 실제 WSL 런타임을 조사했다. 기준 commit은
`5ecc07986`이며 작업 시작 시 worktree는 깨끗했다. 기존 stash 6개는 조회만 했고
적용·삭제·변경하지 않았다.

## 기존 Hermes 확장 지점

- LLM: `agent.auxiliary_client.call_llm`이 provider/model/auth/fallback/retry를 중앙
  관리한다. HEGI는 새 SDK를 추가하지 않고 이 함수를 호출한다.
- Telegram: `tools.send_message_tool._send_telegram`이 기존 Telegram formatter,
  proxy, retry와 Bot API 전송을 제공한다. HEGI는 보고 part별 delivery checkpoint를
  추가하되 실제 송신은 이 경로를 재사용한다.
- MCP: `tools.mcp_tool.discover_mcp_tools`가 설정된 server를 연결하고
  `tools.registry.registry.dispatch`가 동기 handler를 제공한다. 실제 환경에는
  `memory-forest-read.memory_search`와 memory-curator 프로필의
  `memory-forest-curator-draft.memory_create_stm_draft`가 존재한다.
- Config: 행동 설정은 YAML, secret은 `.env`에 둔다. HEGI도 이 경계를 유지하며
  Telegram token 값은 읽되 로그나 상태 DB에 저장하지 않는다.
- Profile: Hermes 경로는 `hermes_constants.get_hermes_home()`을 기준으로 계산한다.
  memory-curator profile에서 실행하면 draft MCP 구성이 해당 profile 범위로 로드된다.
- Test: 저장소 규칙에 따라 모든 테스트는 `scripts/run_tests.sh`로 실행한다.

## 기존 v1 감시기

`~/bin/hegi-memory-watch.py`와 `~/bin/hegi-memory-watch-loop`는 세 DB에서 최근
assistant 발언을 모아 두 명 이상이 응답하고 10분 조용하면 단일 기억 제안 메시지를
보낸다. JSON text 추출, read-only SQLite, 단일 signature 중복 방지는 있으나 전체
논의 구조, Action Item, Memory 검색, archive, delivery checkpoint는 없다.

## 구현 결정

HEGI는 top-level `hegi` Python package로 분리한다. core agent loop, gateway session,
Telegram adapter, MCP server와 Memory Forest 파일을 수정하지 않는다. 수집 cursor와
미처리 message buffer를 별도 SQLite에 유지해 재시작 중 quiet episode가 유실되지
않게 한다. 실제 Telegram 전송은 CLI의 명시적 `--send`가 있어야 한다.

## 확인된 안전 경계

- source DB는 SQLite URI `mode=ro`와 `query_only=ON`으로 연다.
- 설정에서 `auto_commit=true`, `auto_draft=true`,
  `require_professor_approval=false`를 거부한다.
- HEGI 코드에는 Memory Forest commit/write tool 이름이 없다.
- Draft gate는 교수 user ID, 중복되지 않은 승인 message ID, 재검색을 모두 요구한다.
- NAS 미연결 시 local spool을 정본으로 유지하며 기존 archive를 덮어쓰지 않는다.
