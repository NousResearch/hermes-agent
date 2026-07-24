# HEGI v2 — AI Research Secretary

HEGI는 여러 Hermes 프로필의 공진방 대화를 읽기 전용으로 수집해 연구회의 Episode,
한국어 구조화 회의록, Action Item, Memory Evaluation, Markdown/JSON 아카이브와
Telegram 보고로 변환한다.

HEGI는 Hermes 핵심 agent loop나 tool schema를 수정하지 않는 독립 패키지다. LLM,
Telegram, MCP는 Hermes의 기존 호출 경로를 재사용한다. Memory Forest 자동 쓰기와
Commit은 구현하지 않으며, STM Draft는 설정된 교수 계정의 명시적 승인 이후에만
curator draft MCP를 호출한다.

설치와 운영은 [operations.md](docs/operations.md), 안전 경계는
[memory_policy.md](docs/memory_policy.md)를 참고한다.
