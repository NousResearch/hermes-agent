# Implementation Plan

Work Type: feature

## Current State

Desktop voice는 `useVoiceConversation`에서 VAD 녹음 → `/api/audio/transcribe` → 기존 Hermes submit → stream text chunk → `/api/audio/speak` 순서다. 이 구조는 정책 경계가 안전하지만 speech-to-speech Realtime은 아니다. Google Meet plugin에는 OpenAI Realtime WebSocket client가 있으나 beta event names와 단방향 text-to-audio 중심이라 Desktop GA WebRTC 구현의 SSOT로 사용하지 않는다.

## Architecture Contract

`Browser mic → OpenAI WebRTC`는 VAD/transcription 입력만 담당한다. `confirmed transcript → existing Hermes submit/session → gateway events/tool execution → assistant text → existing/streaming TTS`로 이어진다. Realtime response generation과 OpenAI 모델의 Hermes tool schema 접근은 모두 금지한다.

## P0 — Contract Spike and Trust Boundary

Files: test/fixture 및 plan evidence만; production wiring 금지.

구현:
- OpenAI GA `session.update`, transcript, response audio, cancel events의 실제 shape를 fixture로 고정한다.
- Realtime transcription-only session이 response generation 없이 동작하는 GA event/config 계약을 검증한다.
- latency/cost, WebRTC reconnect, browser/Electron support를 측정한다.

완료조건:
- [ ] GA event fixture와 redacted capture 존재.
- [ ] transcript→Hermes→기존 TTS 단일 turn이 spike에서 작동.
- [ ] Realtime response generation 요청이 0건임을 fixture로 확인.

## P1 — Backend Session Broker

Files: Desktop backend 기존 API router, 신규 최소 session helper, focused tests, config schema.

구현:
- authenticated local Desktop 요청만 `/v1/realtime/calls` 또는 `/v1/realtime/client_secrets`를 호출한다.
- standard API key는 process env/provider resolver에서만 읽고 response/log에 넣지 않는다.
- model/voice/TTL/origin/session binding, rate limit, timeout, payload limits를 정의한다.
- feature disabled/missing key/error envelope를 명확히 반환한다.

완료조건:
- [ ] key 비노출 및 auth/origin/rate-limit 테스트 통과.
- [ ] malformed SDP/session config가 fail-closed.
- [ ] session helper는 Hermes tool execution API를 포함하지 않음.

## P2 — Renderer WebRTC Session

Files: `apps/desktop/src/hermes.ts`, 신규 `apps/desktop/src/lib/realtime-voice-session.ts`, voice state/types, focused tests.

구현:
- peer connection, mic track, remote audio, data channel, connection state를 캡슐화한다.
- explicit `connect`, `disconnect`, `cancelInput`, transcript event API를 제공한다.
- all tracks/channel/timers cleanup과 one-session ownership을 보장한다.

완료조건:
- [ ] connect/disconnect/reconnect state tests.
- [ ] secret은 memory-only이며 storage/log 접근이 없음.
- [ ] duplicate listeners/tracks가 발생하지 않음.

## P3 — Hermes Turn Bridge

Files: `use-voice-conversation.ts` 및 기존 message-stream integration의 최소 접점, focused tests.

구현:
- 최종 transcript에 client turn id를 부여해 기존 `onSubmit`을 정확히 1회 호출한다.
- 기존 Hermes assistant stream의 안정 문장 chunk는 기존/streaming TTS 재생 경로로 전달한다.
- tool event는 기존 UI pipeline 그대로 표시하며 Realtime 모델에는 tool 호출 권한을 주지 않는다.
- barge-in은 TTS audio만 취소하고 Hermes tool cancel은 기존 명시적 interrupt에 위임한다.

완료조건:
- [ ] transcript 1 → Hermes turn 1, 중복 0.
- [ ] tool call 1회가 기존 tool UI/event로 관찰됨.
- [ ] barge-in 후 새 turn 가능, 이전 음성 재생 없음.

## P4 — UI, Fallback, Observability

Files: 기존 voice control/settings의 최소 파일, voice docs/tests.

구현:
- `Legacy`와 `Realtime (experimental)` 선택, 연결 상태 pill, mute/end/retry를 제공한다.
- Realtime 실패 시 오류를 표시하고 사용자가 Legacy로 전환할 수 있게 한다.
- secret/audio/transcript 원문을 로그하지 않고 latency/error counters만 기록한다.

완료조건:
- [ ] 기존 legacy mode regression pass.
- [ ] demo와 동일한 상태 전환 및 fallback 동작.
- [ ] config off일 때 Realtime network call 0건.

## P5 — E2E and Rollout

Files: focused E2E fixture/test docs only.

구현:
- mock Realtime peer로 deterministic 10-turn 테스트.
- credential 제공 시 별도 live smoke: 한국어 대화, 1개 read-only Hermes tool, interruption, reconnect.
- experimental/off-by-default release note와 rollback 절차 작성.

완료조건:
- [ ] 10 turns, duplicate 0, secret leak 0, orphan media track 0.
- [ ] `make verify` 또는 repo에 존재하는 동등한 focused verify 통과.
- [ ] live credential 미제공 시 `not verified`를 명시하고 fixture 결과와 분리.

## I/O Contract

| Function | Input | Output | Failure category | Pure |
|---|---|---|---|---|
| createRealtimeSession | authenticated local request + session config | ephemeral secret/SDP envelope | auth/provider/network/schema | No |
| connectRealtimeVoice | ephemeral session + MediaStream | connection state/events | permission/ICE/network | No |
| acceptFinalTranscript | transcript + client turn id | one Hermes submit | duplicate/empty/busy | No |
| speakText | sanitized Hermes text | existing/streaming TTS response | TTS availability/cancel | No |
| reduceRealtimeEvent | event + prior state | next voice state/action | unknown/invalid event | Yes |

## AC → Step Mapping

| AC | Plan Step |
|---|---|
| AC1 | P1, P2 |
| AC2 | P2, P4 |
| AC3 | P0, P3 |
| AC4 | P2, P3 |
| AC5 | P4 |
| AC6 | P1, P2 |
| AC7 | P5 |
