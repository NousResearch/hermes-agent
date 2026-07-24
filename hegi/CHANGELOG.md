# Changelog

## 2.0.0

- 세 Hermes SQLite DB의 공진방 메시지를 읽기 전용 범위 쿼리로 수집한다.
- 중복 user 메시지를 병합하고 assistant identity를 보존한다.
- 침묵·시간 간격·교수 종료 표현을 이용해 Episode를 감지한다.
- Hermes 중앙 LLM 경로로 계층형 구조화 회의록을 생성한다.
- Action Item 중복 방지와 Memory Forest read 평가를 수행한다.
- 승인된 교수 명령 뒤에만 STM Draft MCP를 호출한다.
- revision 보존 Markdown/JSON 아카이브와 checkpointed Telegram 전송을 제공한다.
