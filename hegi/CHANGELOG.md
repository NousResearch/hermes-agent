# Changelog

## 2.0.1

- Memory Curator, Telegram chat/user ID와 agent DB를 자동 탐지하고 즉시 활성화한다.
- 선택한 profile을 LLM·MCP 실행 컨텍스트에도 고정한다.
- readiness, `flock`, stale PID 복구와 Linux/WSL 로그인 자동 시작을 제공한다.
- 실행 중인 gateway를 재시작해 HEGI Telegram pre-dispatch plugin을 반영한다.
- `기억해`·`초안 만들어`를 영속 승인 큐로 선점해 재검색 후 STM Draft만 만든다.
- v1 migration이 watcher 종료, 백업, v2 설치와 daemon 시작까지 수행한다.

## 2.0.0

- 세 Hermes SQLite DB의 공진방 메시지를 읽기 전용 범위 쿼리로 수집한다.
- 중복 user 메시지를 병합하고 assistant identity를 보존한다.
- 침묵·시간 간격·교수 종료 표현을 이용해 Episode를 감지한다.
- Hermes 중앙 LLM 경로로 계층형 구조화 회의록을 생성한다.
- Action Item 중복 방지와 Memory Forest read 평가를 수행한다.
- 승인된 교수 명령 뒤에만 STM Draft MCP를 호출한다.
- revision 보존 Markdown/JSON 아카이브와 checkpointed Telegram 전송을 제공한다.
