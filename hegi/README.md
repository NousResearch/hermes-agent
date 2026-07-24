# HEGI v2.0.1 — AI Research Secretary

HEGI는 여러 Hermes 프로필의 공진방 대화를 읽기 전용으로 수집해 연구회의 Episode,
한국어 구조화 회의록, Action Item, Memory Evaluation, Markdown/JSON 아카이브와
Telegram 보고로 변환한다.

HEGI는 Hermes 핵심 agent loop나 tool schema를 수정하지 않는 독립 패키지다. LLM,
Telegram, MCP는 Hermes의 기존 호출 경로를 재사용한다. Memory Forest 자동 쓰기와
Commit은 구현하지 않으며, STM Draft는 설정된 교수 계정의 명시적 승인 이후에만
curator draft MCP를 호출한다.

`install.sh`은 현재 Hermes 환경에서 Memory Curator profile, Telegram group chat,
교수 user ID와 회의 참여 agent DB를 탐지한다. 비활성 예제 설정을 복사하거나 YAML을
수동 편집하지 않는다.

```bash
cd ~/.hermes/hermes-agent
hegi/scripts/install.sh
python -m hegi doctor
python -m hegi run-once
python -m hegi run-once --send
hegi/scripts/start.sh --send
python -m hegi status
```

교수가 HEGI 회의록에 `기억해` 또는 `초안 만들어`라고 답하면 profile-local
pre-dispatch plugin이 일반 Memory Curator 응답보다 먼저 메시지를 처리한다. 승인
작업은 SQLite에 영속화되며 Memory Forest 재검색 뒤 pending STM Draft만 만든다.

설치와 운영은 [operations.md](docs/operations.md), 안전 경계는
[memory_policy.md](docs/memory_policy.md)를 참고한다.
