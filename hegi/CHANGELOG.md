# Changelog

## 2.0.2

- 교수 메시지의 교차 DB 중복을 제거하고 화자를 `교수`로 정규화한다.
- 연구회의·운영 장애·혼합·기타 meeting type 분류와 운영 장애 전용 템플릿을 추가한다.
- 에이전트 행동 로그와 연구적 의견, 과거·현재 상태 충돌을 분리해 기록한다.
- Memory Evaluation에 검색 결과·중복 대상·신규성 근거를 출력하고 `no_memory`에서
  Draft 생성을 차단한다.
- 최종 quality gate가 raw Python repr, 잘못된 화자명, Memory Forest 직접 쓰기
  제안을 차단한다.

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
