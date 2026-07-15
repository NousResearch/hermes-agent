# Goal

Hermes Desktop에 OpenAI Realtime transcription/VAD 기반 저지연 음성 입력을 선택 기능으로 추가하되, 출력은 기존 또는 streaming TTS를 사용하고 Hermes의 Agent·도구·skill·memory·승인 정책을 유일한 대화 실행 경계로 유지한다.

## Acceptance Criteria

- AC1: Desktop이 backend에서 단기 Realtime client secret/session을 받아 WebRTC로 연결하며 표준 OpenAI API key가 renderer에 노출되지 않는다.
- AC2: 마이크 입력·VAD·transcription은 WebRTC로 흐르고 Hermes 출력은 기존/streaming TTS로 재생되며 연결·청취·생각·도구실행·발화·오류 상태가 UI에 표시된다.
- AC3: Realtime 모델이 직접 Hermes 도구를 실행하지 않는다. 사용자의 확정 transcript가 기존 Hermes session turn으로 제출되고 기존 gateway event/tool pipeline 결과만 음성으로 재생된다.
- AC4: barge-in 시 현재 TTS 재생을 취소하되 이미 시작된 Hermes 도구 실행은 기존 interrupt 정책 없이는 임의 취소하지 않는다.
- AC5: 기존 `STT → Hermes → TTS` 음성모드는 기본값으로 유지되고 Realtime 미설정·실패 시 명시적으로 복귀할 수 있다.
- AC6: session secret TTL, origin/session binding, rate limit, payload size, redacted logging, disconnect cleanup을 테스트한다.
- AC7: 한국어 10-turn E2E에서 transcript/session 연속성, 도구 1회 호출, 중복 turn 0건, secret 노출 0건을 확인한다.

## Non-Goals

- Hermes의 기본 text model을 `gpt-realtime`로 교체하지 않는다.
- Realtime 모델에 Hermes MCP/tool schema 전체를 직접 공개하지 않는다.
- SIP/전화 연결, CLI 실시간 PCM, 모바일 앱은 이번 범위에서 제외한다.
- 기존 Google Meet Realtime client를 그대로 재사용하지 않는다. 단, WebSocket framing·오류처리 패턴은 참고한다.
- 기존 turn-based voice mode를 삭제하지 않는다.

## Stop Conditions

- OpenAI GA event schema를 실환경 smoke로 확인할 수 없거나 API credential 입력이 필요하면 구현은 fixture 계약까지만 진행하고 사용자 입력 대기로 중단한다.
- renderer에 표준 API key가 전달되어야 하는 설계, Hermes 정책 우회, auth/origin 경계 확장이 필요하면 `NEEDS_SCOPE_APPROVAL`.
- 12개 파일 또는 승인된 staged run 범위를 넘으면 플랜 revision이 필요하다.
