# Hermes voice improvements audit

Task: `t_ea62dd92`
Plan source: `/mnt/rackshack/agent-workspace/vault/plans/2026-07-09-hermes-voice-improvements.md`
Branch/worktree: `/home/eric/agent-build/hermes-voice-improvements-t_e06698d9` on `feat/voice-interaction-improvements-t_e06698d9`
Audit date: 2026-07-12

## Scope checked

Reviewed the current branch/worktree for the plan areas called out in the task brief:

- TTS generation and streaming: `tools/tts_tool.py`
- Spoken-output helpers: `hermes_cli/voice_interaction.py`
- CLI voice loop, approval callback, streaming response plumbing: `cli.py`
- STT provider selection and ElevenLabs Scribe path: `tools/transcription_tools.py`, `tools/voice_mode.py`
- Voice defaults: `hermes_cli/config.py`
- Relevant regression tests: `tests/hermes_cli/test_voice_interaction.py`, `tests/tools/test_voice_stt_selection.py`

## Executive summary

The branch already contains a useful first implementation for the CLI voice surface: technical TTS normalization, markdown/code stripping, ElevenLabs expressive-tag gating, pronunciation-dictionary locator passthrough, `voice.stt_provider`, CLI Phase-A acknowledgements, and spoken CLI approval prompts. The remaining gaps are mostly integration depth: acknowledgements are heuristic and CLI-only, ElevenLabs realtime STT is represented as a safe committed-transcript mode rather than a WebSocket realtime implementation, TTS profiles are configured but not fully wired into profile-specific behavior, and barge-in has a playback-stop primitive but not full user-speech interruption with history marking.

## Acceptance-criteria map

| Plan acceptance criterion | Current status | Evidence / current files | Files/modules likely needing changes |
|---|---|---|---|
| Hermes can speak a short acknowledgement before slow tool calls. | Partially implemented. CLI voice mode calls `choose_voice_acknowledgement()` before `run_conversation()`, so the ack is scheduled before the agent can enter tool execution. The decision is heuristic and does not inspect the actual model tool-call decision. | `cli.py:12300-12309`; `hermes_cli/voice_interaction.py:254-272`; tests in `tests/hermes_cli/test_voice_interaction.py:50-53`. | For robust behavior: `cli.py` and/or agent/tool-dispatch hooks in `run_agent.py` / `model_tools.py` so acknowledgements can be based on actual pending tool calls and can cover non-CLI voice surfaces. |
| Hermes no longer silently waits through common tool/delegation delays. | Partially implemented. The pre-agent ack covers likely slow user text in CLI voice mode. It does not emit meaningful progress while long commands/delegations are running, and it is not wired to all gateway/Discord voice-channel paths. | `cli.py:12300-12309`; tool-progress callback area starts around `cli.py:10935`. | `cli.py` tool-progress callback; gateway voice reply pipeline; Discord voice mixer / platform adapters if chief Discord voice is in scope. |
| Technical acronyms, IPs, ports, VLANs, domains, and service accounts are normalized before TTS. | Implemented for the checked examples and many common cases. | `hermes_cli/voice_interaction.py:31-42`, `121-159`, `239-251`; `tools/tts_tool.py:2203-2207` applies it to the TTS tool; CLI final speech also calls it at `cli.py:11363-11370`; tests in `tests/hermes_cli/test_voice_interaction.py:10-30`. | Continue extending `hermes_cli/voice_interaction.py` as new pronunciation failures appear. |
| Expressive tags are supported only when enabled and only for compatible ElevenLabs models. | Implemented locally. Tags are disabled by default, model-gated by allowlist, and limited to one whitelisted tag. | `hermes_cli/config.py:2147-2152`; `hermes_cli/voice_interaction.py:18-20`, `206-236`; tests in `tests/hermes_cli/test_voice_interaction.py:41-48`. | Product decision still needed for which ElevenLabs model IDs belong in the default allowlist; current default is empty. |
| Unsupported expressive tags are stripped before TTS. | Implemented locally. Unknown bracket tags and extra supported tags are stripped unless enabled and allowlisted. | `hermes_cli/voice_interaction.py:224-236`; `tools/tts_tool.py:2203-2207`; tests in `tests/hermes_cli/test_voice_interaction.py:41-48`. | None for first pass; add provider-specific tests if non-ElevenLabs providers should preserve/strip differently. |
| Voice output avoids markdown, raw JSON, tables, and command spam. | Partially implemented. `prepare_spoken_text()` strips code fences, tables, markdown links/emphasis, large JSON-ish blocks, and normalizes final TTS text. It is applied by the TTS tool and CLI final speech. | `hermes_cli/voice_interaction.py:161-176`, `239-251`; `tools/tts_tool.py:2203-2207`; `cli.py:11363-11380`; tests in `tests/hermes_cli/test_voice_interaction.py:33-39`. | More natural summarization may need an LLM rewrite layer rather than regex stripping, especially for long stack traces, large JSON/YAML, and command output. |
| Local Whisper and ElevenLabs STT can be selected by config. | Implemented for config selection. `voice.stt_provider` aliases `local_whisper`, `elevenlabs_scribe`, and `elevenlabs_scribe_realtime` onto the STT dispatcher. | `hermes_cli/config.py:2145`; `tools/transcription_tools.py:121-143`, `256-265`, `843-845`, `1714-1722`; tests in `tests/tools/test_voice_stt_selection.py:4-13`. | Docs/setup UI may need updates so users discover the new `voice.stt_provider` choices. |
| ElevenLabs realtime STT can use partial transcripts safely without executing actions from partial text. | Mostly blocked / not truly implemented. The realtime value is accepted, but the implementation uses the existing HTTP Scribe transcription path and annotates results as committed-only with `partial_transcripts_executed = False`. This is safe, but it is not WebSocket realtime Scribe and does not consume partial transcripts. | `tools/transcription_tools.py:124-128`, `1714-1722`; test in `tests/tools/test_voice_stt_selection.py:16-29`. | New realtime WebSocket STT module, async session lifecycle, partial-transcript UI/anticipation path, and hard guard that only final/committed segments enter `run_conversation()`. Also confirm ElevenLabs API capability/details before implementation. |
| TTS playback can be interrupted by new user speech. | Partially implemented. There is an interruptible playback primitive (`stop_playback()`) and `/voice off` stops active TTS. Full barge-in is not complete: the recorder/listener is not clearly active while TTS is speaking, queued audio cancellation is limited to current playback/stream stop events, and history is not marked as interrupted spoken output. | `tools/voice_mode.py:1014-1052`; `cli.py:11483-11511`; streaming TTS honors `stop_event` in `tools/tts_tool.py:2703-2748`. | `cli.py` continuous voice loop, `tools/voice_mode.py`, `tools/tts_tool.py`, and conversation-history metadata in `run_agent.py` / CLI history handling. Need a design decision for how to mark “interrupted, not fully delivered” in persisted messages. |
| Approval prompts are spoken immediately and clearly. | Implemented for CLI dangerous-command approvals. The modal is set up, then voice TTS says “I need approval to run that command.” while tool execution remains blocked on the approval queue. The broader “after approval: acknowledge briefly” and denial spoken summary are not yet implemented. | `cli.py:11696-11716`, `11727-11756`; heuristic approval ack in `hermes_cli/voice_interaction.py:254-264`. | `cli.py` approval callback for post-approval / denied spoken outcomes; gateway approval flow if voice approvals outside CLI are required. |

## Core behavior requirement map

1. Immediate spoken acknowledgement before slow work: partial, CLI-only heuristic. Needs actual tool-call / slow-operation integration and likely gateway coverage.
2. Split voice output into Phase A acknowledgement and Phase B final answer: partial. CLI has separate ack scheduling plus final TTS, but no shared event abstraction and no global gateway voice phase boundary.
3. TTS text-normalization layer: implemented in `hermes_cli/voice_interaction.py` and applied in the TTS tool.
4. Rewrite final answers for spoken language: partial. Regex cleanup and voice prompt are present; no true summarizing rewrite layer for very structured output.
5. Controlled ElevenLabs expressive tags: implemented locally, default disabled and allowlist-gated; default allowlist remains a product/config decision.
6. Optional pronunciation dictionary support: partial. Locator IDs are accepted and sent to ElevenLabs, but no bundled pronunciation dictionary/map artifact exists beyond normalization terms.
7. Model/profile selection for TTS: partial. Default profiles exist in config, and `resolve_voice_tts_profile()` applies `normalize` / tag policy. The `streaming` profile key is not fully wired as a central selector for the CLI/gateway generation path.
8. STT backend abstraction: partial. Config can select local Whisper and ElevenLabs Scribe variants; realtime WebSocket is not implemented.
9. Interruption / barge-in support: partial. Playback can be stopped, but automatic barge-in on new speech and interrupted-history preservation remain to design/implement.
10. Approval-aware voice behavior: partial. Immediate spoken approval request exists for CLI command approvals; post-approval / denied spoken responses are missing.
11. Spoken-output testing examples: implemented for the plan’s four normalization examples, plus markdown stripping, tag gating, pronunciation locators, profile behavior, and STT selection.

## Product / capability decisions to resolve before implementation cards

- Should this improvement target only CLI `/voice` first, or also gateway voice replies and Discord voice-channel mode used by `systems/chief-discord-voice`?
- Which ElevenLabs TTS model IDs are officially allowed for expressive tags by default, if any? Current safe default is an empty allowlist.
- Should `elevenlabs_scribe_realtime` be a true WebSocket streaming backend now, or remain a safe alias to committed HTTP Scribe until realtime UX is designed?
- How should Hermes persist an assistant response that was generated but only partially spoken before barge-in? Current history code has interruption concepts for agent turns, but not a clear spoken-delivery marker.
- Should spoken-output rewriting stay deterministic/regex-only, or use an auxiliary LLM pass for long structured answers before TTS?

## Suggested next implementation cards

1. Wire `voice.tts_profile` into a single TTS policy object used by CLI, gateway, and streaming TTS; make `streaming` choose the generation path.
2. Move Phase-A acknowledgement from a CLI-only heuristic into a voice event layer that can fire before first tool execution and before approval waits across surfaces.
3. Implement or defer true ElevenLabs Scribe Realtime WebSocket support; if implemented, keep partial transcripts display/anticipation-only and commit only final segments.
4. Implement full barge-in: listen during playback, stop current/queued audio, mark spoken delivery as interrupted, and start the new user turn.
5. Add docs/config examples for `voice.stt_provider`, `voice.tts_profile`, expressive tags, and ElevenLabs pronunciation dictionaries.

## Branch readiness

The isolated branch/worktree already exists and is suitable for follow-up implementation work:

- Worktree path: `/home/eric/agent-build/hermes-voice-improvements-t_e06698d9`
- Branch: `feat/voice-interaction-improvements-t_e06698d9`
- Audit artifact: `notes/voice-audit.md`
