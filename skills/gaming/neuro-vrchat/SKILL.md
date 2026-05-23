---
name: neuro-vrchat
description: "Bridge Neuro API actions into safe VRChat autonomy."
version: 1.0.0
author: Bob Nyan, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [vrchat, neuro-api, osc, voicevox, autonomy, gaming]
    category: gaming
    related_skills: [vrchat-osc]
---

# Neuro VRChat Bridge

Use this skill when a user wants Hermes to expose a Neuro API-compatible bridge for VRChat autonomy.

## Prerequisites

- A local `vendor/neuro-sdk` clone from `https://github.com/VedalAI/neuro-sdk`.
- VRChat running with official OSC enabled.
- `python-osc` available through the `vrchat` extra.
- VOICEVOX Engine on `http://127.0.0.1:50021` when speech is enabled.
- A local VRChat autonomy profile. Missing profiles are treated as disabled and dry-run.
- `websockets==15.0.1` when running `scripts/vrchat_neuro_bridge.py`.

## Operating Rules

1. Use `vrchat_neuro_status` before attempting bridge work.
2. Use `vrchat_neuro_build_messages` to prepare `startup`, optional `context`, and `actions/register` messages.
3. Use `vrchat_neuro_handle_action` for each incoming Neuro `action` message.
4. Use `vrchat_observation_ingest` for STT, vision, stream, operator, or system observations.
5. Use `vrchat_observation_from_osc` for incoming VRChat OSC ChatBox events.
6. Use `vrchat_observation_queue_status` before a heartbeat or scheduler consumes queued context.
7. Treat Neuro action data as untrusted input. Invalid JSON, unknown actions, raw OSC, disabled profile output, and unknown avatar actions must be rejected.
8. Keep the profile disabled or `dry_run: true` until the operator intentionally tests in a private VRChat instance.
9. Live OSC or audio is allowed only through the existing VRChat autonomy profile safety gate, never through raw Neuro action payloads.
10. Use `vrchat_autonomy_prepare_profile` to create the local private-test dry-run profile before heartbeat work.
11. Use `vrchat_autonomy_profile_status` and `vrchat_autonomy_profile_tick` for profile-driven heartbeat or scheduler checks.
12. Use `vrchat_autonomy_heartbeat_tick` when a scheduler should turn a ready launch/readiness event into one profile-driven tick.
13. Use `vrchat_autonomy_conversation_dry_run` to prove vision, STT, ChatBox, and operator observations can plan ChatBox/VOICEVOX output and route through Neuro without live actuation.
14. Use `vrchat_autonomy_preflight_bundle` before any private-instance live test, and inspect `vrchat_process.phase`, `voicevox.process.phase`, `voicevox_synthesis`, and `audio.virtual_cable_route` when the operator says VRChat or VOICEVOX was started but readiness is still blocked.
15. Use `vrchat_autonomy_runtime_doctor` when operator-reported runtime state disagrees with read-only readiness; inspect its VOICEVOX URL probes, port snapshot, visible-window evidence, process visibility, and bounded launch discovery before asking the operator to retry.
16. Use `vrchat_autonomy_wait_ready` when the operator is actively launching VRChat or VOICEVOX and wants a bounded read-only wait loop.
17. Use `vrchat_autonomy_wait_then_tick` when the operator wants readiness wait followed by one gated profile tick.
18. Use `vrchat_autonomy_prepare_private_smoke` immediately before any live private smoke to evaluate live gates and build a dry-run plan without live output.
19. Use `vrchat_autonomy_wait_then_private_smoke` when the operator has launched VRChat and VOICEVOX and wants the harness to wait until readiness, then stop at preparation by default.
20. Use `vrchat_autonomy_private_smoke` before any private-instance live test.
21. Use `vrchat_autonomy_completion_audit` before claiming the full VRChat autonomy objective is complete.

## Harness

The Neuro harness is `scripts/vrchat_neuro_bridge.py`. It opens a websocket to a Neuro API server, sends bootstrap messages, receives `action` commands, routes them through `vrchat_neuro_handle_action`, and returns `action/result`.

The harness does not bypass the Hermes profile. If the profile is missing, invalid, disabled, or dry-run, no live VRChat OSC or VOICEVOX audio should be produced.

The observation harness is `scripts/vrchat_observation_harness.py`. It queues JSONL events from STT, vision, stream chat, or operator panels, and can also listen for incoming `/chatbox/input` OSC events. Its `--tick-profile` mode refuses live profiles unless `--allow-live-profile` is supplied.

The heartbeat tick harness is `scripts/vrchat_heartbeat_tick.py`. It first runs the read-only heartbeat, then runs one profile tick only on `VRCHAT_LAUNCHED_READY` or `READINESS_COMPLETE` unless the operator asks for `--tick-when-already-ready` or `--force-tick`. It refuses non-dry-run profiles unless `--allow-live-profile` and the exact live acknowledgement are both supplied.

The conversation dry-run harness is `scripts/vrchat_conversation_dry_run.py`. It runs representative multimodal observations through local planning and the Neuro action path, defaults to non-persistent observations, and must keep all actuation safety flags false.

The profile harness is `scripts/vrchat_profile.py`. It prepares an enabled private-test dry-run profile by default, with VOICEVOX and ChatBox allowed, movement blocked, the virtual cable playback side set to `CABLE Input`, and the VRChat microphone side set to `CABLE Output`. Its `--arm-live` mode refuses to write unless the exact live acknowledgement is supplied.

The preflight harness is `scripts/vrchat_preflight.py`. It collects profile, readiness, Neuro SDK vendor, observation queue, audio output device evidence, the virtual cable playback/microphone-side route, and optional no-playback VOICEVOX synthesis without sending OSC, playing audio, recording microphone input, or opening a Neuro websocket.

The runtime doctor harness is `scripts/vrchat_runtime_doctor.py`. It extends preflight with operator mismatch flags, common VOICEVOX local URL probes, local port snapshots, relevant visible Windows desktop windows, bounded process visibility diagnostics without storing command lines, bounded read-only launch discovery for Steam VRChat and common VOICEVOX locations, and concrete next actions. It is read-only, does not launch apps, and must keep all live actuation flags false.

The private smoke preparation harness is `scripts/vrchat_private_smoke.py --prepare-only`. It validates readiness, profile state, and live acknowledgement, then builds a dry-run ChatBox/VOICEVOX/avatar action plan. It must never send ChatBox, play audio, or write avatar parameters.

The wait harness is `scripts/vrchat_wait_ready.py`. It repeatedly collects the same read-only preflight evidence until readiness succeeds or timeout expires. It records bounded snapshots and never runs a profile tick by itself.

The wait-then-tick harness is `scripts/vrchat_wait_then_tick.py`. It waits for readiness, then calls the heartbeat tick path with `tick_when_already_ready`. It keeps the profile gate intact; non-dry-run profiles still require `--allow-live-profile` and the exact live acknowledgement.

The wait-then-private-smoke harness is `scripts/vrchat_wait_then_private_smoke.py`. It waits for read-only readiness, then runs the private smoke preparation path. It must not run live output unless `--allow-live-smoke`, complete readiness, a valid armed non-dry-run profile, and the exact live acknowledgement are all present.

The private smoke harness is `scripts/vrchat_private_smoke.py`. It defaults to dry-run and requires readiness, a valid enabled non-dry-run profile, and the exact live acknowledgement before any ChatBox, VOICEVOX, or avatar action execution.

The completion audit harness is `scripts/vrchat_completion_audit.py`. It is read-only and reports whether the current workspace/runtime evidence satisfies the full objective. It verifies a dry-run multimodal turn plan from synthetic observation context, no-playback VOICEVOX synthesis, and one synthetic Neuro action through the Hermes safety gate without live output. It must leave live smoke incomplete until the local runtime is ready and the operator deliberately arms live output.

## Action Surface

Register only these Hermes-backed actions:

- `vrchat_autonomy_turn`
- `vrchat_speak`
- `vrchat_chatbox`
- `vrchat_avatar_action` when the profile has allowed avatar actions

Avatar actions must come from profile action IDs. Do not expose OSC addresses, OSC args, VRChat credentials, client modification paths, or arbitrary code execution as Neuro actions.

## Verification

For local validation, run focused compile, unit tests, registry discovery, and a read-only readiness check. A manual live smoke test requires VRChat, VOICEVOX, OSC enabled, the virtual cable selected in VRChat, and an explicit non-dry-run profile acknowledgement.
