# Status

- Lifecycle: `implemented_verified_off_by_default`
- Revision: `v3`
- Work size/risk: `L / R3`
- Implementation: complete for experimental/off-by-default mode
- GPT Pro research: complete, original full-duplex proposal rejected; hybrid Realtime STT/VAD + Hermes + TTS adopted
- OpenAI official docs checked: Realtime GA, WebRTC, `/v1/realtime/calls`, `/v1/realtime/client_secrets`
- Hermes code state checked with CodeGraph: existing Desktop voice is turn-based STT/Hermes/TTS; Google Meet has a legacy server WebSocket Realtime client but uses pre-GA event names.
- Dirty tree: Hermes repo already has unrelated modified/untracked files. Implementation must use scoped diff/status and must not claim a clean repository.
- Worker preflight corrected the backend owner from nonexistent `apps/desktop/backend/**` to `hermes_cli/web_server.py`; same feature scope, no behavioral expansion.

## Review Gates

- [ ] GPT Pro research: architecture, trust boundary, interruption semantics, cost/latency
- [ ] GPT Pro plan review: AC-step-test causality and security blockers
- [ ] OpenAI GA event schema fixture verified
- [ ] User approves implementation scope after review

## Known Risks

- Realtime model과 Hermes Agent가 모두 대화를 생성하면 이중 agent/중복 응답이 발생할 수 있다. 본 플랜은 Realtime을 음성 transport로 제한한다.
- WebRTC output을 Hermes 결과로 자연스럽게 발화시키는 방식은 GA event 계약을 spike로 검증해야 한다.
- `plugins/google_meet/realtime/openai_client.py`는 `response.audio.delta` 등 beta 이벤트를 사용하므로 직접 재사용하면 안 된다.

## Next Action

실제 OpenAI credential과 Desktop mic/ICE 환경에서 opt-in live smoke를 별도 수행한다. 현재 deterministic tests/build는 통과했으며 live smoke는 NOT VERIFIED다.
