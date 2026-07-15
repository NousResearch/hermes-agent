# Wiki / Retrieval Context

## Sources Consulted

- Hermes 공식 Configuration 문서: `/voice on`, STT provider, TTS spoken replies가 현재 지원됨.
- OpenAI 공식 Realtime API 문서: `gpt-realtime`, WebRTC/WebSocket/SIP, GA event model, client secret endpoint.
- OpenAI 공식 WebRTC 문서: client에는 WebRTC 권장, 표준 API key는 backend에만 보관.
- CodeGraph `/home/p/.hermes/hermes-agent`:
  - `apps/desktop/src/hermes.ts::transcribeAudio/speakText`
  - `apps/desktop/src/app/chat/composer/hooks/use-voice-conversation.ts::useVoiceConversation`
  - `apps/desktop/src/lib/voice-playback.ts::playSpeechText`
  - `agent/transcription_provider.py::TranscriptionProvider`
  - `agent/tts_provider.py::TTSProvider`
  - `plugins/google_meet/realtime/openai_client.py` (legacy/beta reference only)
- Vault semantic search: 직접적인 Realtime voice ADR은 없었음. 관련 결정으로 `projects/ai-coding-brain-mcp/wiki/decisions/adr-016-orchestra-gateway-mcp-tools.md`가 Hermes gateway를 실행 경계로 두는 맥락을 제공함.

## Decisions

1. OpenAI Realtime은 transcription/VAD 입력 transport로만 사용하고 response generation에는 사용하지 않는다.
2. Hermes의 기존 session, tool execution, skill/wiki/memory, 승인 정책이 SSOT다.
3. renderer에는 단기 client secret만 전달하고 표준 API key는 backend 밖으로 내보내지 않는다.
4. 기존 legacy voice mode를 기본 및 rollback 경로로 유지한다.
5. Hermes 출력은 기존 또는 streaming TTS를 사용해 의미 변형과 이중 conversation을 차단한다.
6. browser/Desktop은 WebRTC, 서버측 CLI 확장은 후속 플랜으로 분리한다.

## Constraint / Verify Against

- Realtime direct tool execution 금지.
- transcript 1건당 Hermes turn 1건.
- OpenAI key와 ephemeral secret의 로그/저장 노출 0건.
- beta event names를 GA 구현에 복사하지 않음.
- 실패를 숨기는 자동 fallback 대신 사용자에게 상태와 선택권 제공.

## Unknown or Drift

- OpenAI voice/tool/session event 세부 shape와 지원 model은 구현 직전 공식 API reference와 live fixture로 재확인한다.
- Hermes Desktop backend API router의 정확한 변경 파일은 CodeGraph 결과가 좁히지 못했으므로 Run 1 preflight에서 확정한다.
- Vault에는 이 기능 전용 ADR이 없으므로 구현 승인 시 Hermes repo 내 design note 또는 project wiki ADR 후보를 검토한다.

## 과거 학습 (get_related_lessons)

Hermes repo에는 `docs/plans/_meta/plan_lessons.jsonl`이 없어 관련 lesson injection을 수행할 수 없었다. 대체 근거로 CodeGraph, 공식 문서, 활성 vault ADR을 사용했다.

## Closeout Learning Targets

- Realtime voice를 agent가 아닌 transport로 제한할 때의 turn ownership 계약.
- WebRTC disconnect/barge-in과 Hermes tool execution의 취소 경계.
- client secret 발급 endpoint의 최소 보안·관측 계약.
