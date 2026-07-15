# Test Plan

## Primary Verification

```bash
make verify
```

Hermes repo에 `make verify`가 없거나 전체 baseline이 unrelated dirty-tree noise로 실패하면, Worker는 변경 파일에 해당하는 기존 package scripts와 focused pytest/Jest 명령을 기록하고 Hermes가 동일 명령을 재실행한다.

## Test Matrix

| Unit | Success | Boundary | Failure |
|---|---|---|---|
| Session broker | ephemeral response, key redacted | TTL/model/voice limits | missing key, bad auth, upstream timeout |
| Event reducer | GA events transition correctly | unknown/duplicate/out-of-order | malformed payload fail-safe |
| WebRTC session | connect/audio/data/disconnect | reconnect, mic denied | ICE fail, channel close, timeout |
| Turn bridge | transcript exactly-once submit | empty/rapid/barge-in | duplicate id, busy session |
| Speech bridge | Hermes text sent to existing/streaming TTS in order | chunk boundary, Korean punctuation | TTS error/cancel |
| Fallback UI | explicit Legacy switch | config disabled | Realtime failure visible |
| Security | no standard key in renderer/log | ephemeral memory-only | storage/log leak test fails |

## Required Focused Tests

1. Backend contract tests with mocked OpenAI endpoints; assert request auth and response redaction.
2. Renderer unit tests with mocked `RTCPeerConnection`, `MediaStream`, data channel and audio track.
3. `useVoiceConversation` regression tests: legacy path unchanged; Realtime exactly-once transcript and cancellation.
4. Fixture test rejecting beta names and asserting Realtime response generation events are never sent.
5. Cleanup test asserting tracks stopped, peer/data channel closed, timers/listeners removed.
6. 10-turn deterministic E2E with one mocked Hermes tool event.

## AC → Verification

| AC | Verification |
|---|---|
| AC1 | broker auth/redaction tests + renderer bundle/storage scan |
| AC2 | mocked WebRTC media/state test + UI interaction test |
| AC3 | no direct tool schema assertion + existing Hermes tool event E2E |
| AC4 | barge-in cancel/cleanup test; Hermes execution remains active unless explicit interrupt |
| AC5 | legacy regression + config-off network-call-zero test |
| AC6 | TTL/origin/rate/payload/log/cleanup tests |
| AC7 | 10-turn fixture metrics; optional credentialed live smoke reported separately |

## Falsifiable Release Gates

- Any standard API key string reaches renderer payload, log, storage or snapshot → FAIL.
- Any final transcript produces 0 or >1 Hermes submit → FAIL.
- Any Realtime response generation or function/tool call occurs → FAIL.
- disconnect leaves an active audio track/channel/timer → FAIL.
- legacy voice focused test regression → FAIL.
- credential 없는 환경에서 live test를 PASS로 표기 → FAIL.
