# VRChat Neuro-sama Style Autonomy

This fork exposes a safe Hermes-native path for a Neuro-sama-style VRChat avatar loop.

The implementation is intentionally conservative:

- It uses the official VRChat client and official OSC only.
- It does not log in to VRChat or read VRChat credentials.
- It does not modify the VRChat client.
- It does not send raw model-selected OSC.
- It keeps ChatBox, VOICEVOX speech, and avatar actions behind explicit validation gates.
- It defaults autonomous turns to dry-run.

## Local Stack

Required local pieces:

- VRChat with OSC enabled from the Action Menu.
- `python-osc` from the `vrchat` extra.
- VOICEVOX Engine on `http://127.0.0.1:50021`.
- Optional virtual audio cable, such as VB-CABLE, for routing VOICEVOX audio into VRChat's microphone input.
- Optional Hypura harness on `http://127.0.0.1:18794`.

VRChat's default OSC ports are:

- Send into VRChat: `127.0.0.1:9000`
- Listen to VRChat: `127.0.0.1:9001`

## Readiness

The `vrchat_autonomy_status` tool performs a read-only readiness check:

- VRChat process is visible.
- `python-osc` is importable.
- VOICEVOX `/version` responds.
- Hypura harness `/status` responds when required.
- A configured audio output device can be found when supplied.

When VRChat is not yet ready, the `vrchat_process` check also reports a
diagnostic phase:

- `running` when `VRChat.exe` is visible.
- `launching_or_blocked` when a VRChat launcher or Easy Anti-Cheat related process is visible, but `VRChat.exe` is not.
- `steam_running_no_vrchat` when Steam is running without a visible VRChat process.
- `not_detected` when no relevant local process is visible.

Only `running` satisfies readiness. Launcher clues are evidence for operator
diagnosis, not permission to send OSC or audio.

The VOICEVOX check also includes a process diagnostic. HTTP `/version` must
respond before speech is considered ready, but the process phase can explain
operator-facing mismatches:

- `engine_process_running` when the local Engine process is visible.
- `ui_running_no_engine` when the VOICEVOX UI is visible but the Engine process is not.
- `not_detected` when no VOICEVOX UI or Engine process is visible.

This readiness check does not:

- send OSC,
- play audio,
- record microphone input,
- write avatar parameters,
- change VRChat settings.

The `vrchat_autonomy_heartbeat` tool adds launch/change detection on top of
readiness. It persists a small local state file under Hermes home and returns
`notify: false` with `HEARTBEAT_OK` when VRChat is not running or nothing
actionable changed.

Notification-style events include:

- `VRCHAT_LAUNCHED_READY`
- `VRCHAT_LAUNCHED_BLOCKED`
- `READINESS_COMPLETE`
- `READINESS_CHANGED`

For manual startup checks, `vrchat_autonomy_wait_ready` and
`scripts/vrchat_wait_ready.py` poll the same read-only readiness bundle until
all prerequisites are ready or a timeout expires. The wait harness records
bounded snapshots with VRChat and VOICEVOX process phases, but it does not run
a profile tick, send OSC, play audio, record the microphone, or open a Neuro
websocket.

When the operator wants the first safe turn to happen immediately after
readiness completes, use `vrchat_autonomy_wait_then_tick` or
`scripts/vrchat_wait_then_tick.py`. It waits with the same read-only checks,
then calls `vrchat_autonomy_heartbeat_tick` with `tick_when_already_ready`.
Live profiles still require `--allow-live-profile` plus the exact live
acknowledgement, and dry-run profiles remain the default.

For integration proof without live VRChat or VOICEVOX output,
`vrchat_autonomy_conversation_dry_run` and
`scripts/vrchat_conversation_dry_run.py` run representative vision, STT,
ChatBox, and operator observations through normalization, dry-run
ChatBox/VOICEVOX planning, and the Neuro action route. The default mode does
not persist observations and does not send OSC, play audio, record microphone
input, or open a Neuro websocket.

When the operator reports that VRChat or VOICEVOX is already running but
readiness remains blocked, use `vrchat_autonomy_runtime_doctor` or
`scripts/vrchat_runtime_doctor.py`. It wraps the preflight bundle, probes common
VOICEVOX local URLs, snapshots relevant local ports, records relevant visible
Windows desktop windows, performs bounded read-only launch discovery for Steam
VRChat and common VOICEVOX install or shortcut locations, records bounded
process visibility diagnostics without storing command lines, records
operator/runtime mismatches, and returns concrete next actions. It remains
read-only and does not launch apps, send OSC, play audio, record microphone
input, open a Neuro websocket, or arm a live profile.

## Virtual Cable Voice Routing

Install a virtual cable and select its recording side as the microphone input inside VRChat. Then route VOICEVOX playback to the matching playback side.

For VB-CABLE on Windows, this usually means:

- Hermes / VOICEVOX playback target: `CABLE Input (VB-Audio Virtual Cable)`
- VRChat microphone device: `CABLE Output (VB-Audio Virtual Cable)`

Hermes does not change VRChat's selected microphone. The operator must choose the correct device in VRChat.
The preflight bundle now records both sides under
`audio.virtual_cable_route`: the VOICEVOX playback side and the VRChat
microphone side. It only enumerates devices; it does not record microphone
input or play audio. If the profile contains only `audio_output_device`,
Hermes infers the microphone-side name from common cable naming, for example
`CABLE Input` to `CABLE Output`.

For stronger speech readiness evidence, the preflight can also run a
no-playback VOICEVOX synthesis probe. This sends `audio_query` and `synthesis`
to the local Engine and records only metadata such as `size_bytes` and
`wav_header_ok` under `voicevox_synthesis`; it does not route the WAV to any
audio device.

`voicevox_speak` accepts an optional `output_device` field:

```json
{
  "text": "konnichiwa",
  "speaker": 8,
  "blocking": true,
  "output_device": "CABLE Input"
}
```

When `output_device` is not provided, the existing platform playback path is used.

## Observations

Autonomous turns consume bounded observation events. Supported sources are:

- `textBox`
- `speechToText`
- `visionObservation`
- `streamComment`
- `operator`
- `system`

Example:

```json
[
  {"source": "speechToText", "text": "hello"},
  {"source": "visionObservation", "summary": "The user is waving."},
  {"source": "streamComment", "content": "Nice move"}
]
```

Visual input should be summarized before it reaches this tool. The autonomy layer treats multimodal perception as context, not as a direct actuator command.

Use `vrchat_autonomy_enqueue_observation` when an external listener, STT
pipeline, vision summarizer, stream-chat bridge, or operator action wants to
hand one event to the periodic loop:

```json
{
  "observation": {
    "source": "visionObservation",
    "summary": "The user is waving."
  }
}
```

The queue is stored under Hermes home as a JSONL state file. Invalid or unknown
sources are rejected before they reach the model.

Use `vrchat_observation_ingest` for batch handoff from an STT process, a vision
summarizer, a stream chat bridge, or an operator panel:

```json
{
  "observations": [
    {"source": "speechToText", "text": "hello"},
    {"source": "visionObservation", "summary": "The user is waving."},
    {"source": "streamComment", "content": "Nice move"}
  ]
}
```

Use `vrchat_observation_from_osc` when an OSC listener receives a single VRChat
event. `/chatbox/input` is converted into `textBox` context. Avatar parameter
events are ignored by default and can only be admitted as `system` observations
when `allow_avatar_parameters` is explicitly true.

`vrchat_observation_queue_status` previews the persisted queue without
consuming it.

The optional local harness is `scripts/vrchat_observation_harness.py`:

```powershell
py -3.12 scripts\vrchat_observation_harness.py --stdin-jsonl
py -3.12 scripts\vrchat_observation_harness.py --listen-osc --osc-port 9001
```

JSONL stdin accepts events such as:

```json
{"source": "visionObservation", "summary": "The user is waving."}
```

With `--tick-profile`, the harness can run one profile tick after queueing an
accepted observation. It refuses live profiles by default; `--allow-live-profile`
is required when the profile has `dry_run: false`.

## Structured Decision

The model decision must be structured. Raw OSC is rejected.

Allowed fields:

```json
{
  "speak_text": "short spoken line",
  "chatbox_text": "short chatbox line",
  "emotion": "happy",
  "avatar_action": "wave",
  "urgency": "low"
}
```

Safety limits:

- ChatBox max: 144 characters and 9 lines.
- Speech max: 200 characters.
- Raw OSC addresses are rejected.
- Unknown avatar actions are rejected.
- Public mode blocks movement-like actions.
- Critical interruption is blocked unless explicitly enabled.

## Decision Request Builder

Use `vrchat_autonomy_build_decision_request` to turn normalized observations
and currently allowed actions into a JSON-schema LLM request. The tool does not
call an LLM. It returns messages and a strict schema that another Hermes model
call can use to produce the decision.

```json
{
  "observations": [
    {"source": "speechToText", "text": "hello"},
    {"source": "visionObservation", "summary": "The user is waving."}
  ],
  "mode": "private_test",
  "allow_voice": true,
  "allow_chatbox": true,
  "allowed_avatar_actions": ["wave"],
  "avatar_action_descriptions": {
    "wave": "Wave once."
  }
}
```

## Model-Calling Turn

Use `vrchat_autonomy_run_turn` when Hermes should perform the full single-turn
loop:

1. Normalize observations.
2. Build the structured decision request.
3. Call the configured `auxiliary.vrchat_autonomy` model.
4. Parse strict JSON.
5. Validate the untrusted decision.
6. Plan or execute only allowed actions.

It still defaults to `dry_run: true`, so the first result should be an audit
plan, not VRChat actuation.

```json
{
  "observations": [
    {"source": "speechToText", "text": "hello"},
    {"source": "visionObservation", "summary": "The user is waving."}
  ],
  "mode": "private_test",
  "allow_voice": true,
  "allow_chatbox": true,
  "dry_run": true,
  "output_device": "CABLE Input",
  "avatar_action_profiles": {
    "wave": [{"name": "Wave", "value": true}]
  },
  "avatar_action_descriptions": {
    "wave": "Wave once."
  }
}
```

Optional model routing in `config.yaml`:

```yaml
auxiliary:
  vrchat_autonomy:
    provider: auto
    model: ""
    timeout: 60
    extra_body: {}
```

## Periodic Loop Tick

Use `vrchat_autonomy_loop_tick` for a safe scheduler or heartbeat-driven
single iteration. It is not a hidden daemon. Each call:

1. Honors `emergency_stop`.
2. Refuses to proceed unless `enabled: true`.
3. Checks VRChat, `python-osc`, VOICEVOX, optional harness, and optional audio output readiness.
4. Enforces `min_turn_interval_sec`.
5. Reads explicit observations and the persisted observation queue.
6. Calls `vrchat_autonomy_run_turn`.
7. Persists a small loop state summary under Hermes home.

Dry-run is still true by default:

```json
{
  "enabled": true,
  "mode": "private_test",
  "allow_voice": true,
  "allow_chatbox": true,
  "dry_run": true,
  "min_turn_interval_sec": 10,
  "output_device": "CABLE Input",
  "avatar_action_profiles": {
    "wave": [{"name": "Wave", "value": true}]
  },
  "avatar_action_descriptions": {
    "wave": "Wave once."
  }
}
```

Queued observations are consumed only after the model response reaches local
decision validation and planning. A transient model-call failure keeps the queue
for a later tick.

Emergency stop is explicit and non-actuating:

```json
{
  "enabled": true,
  "emergency_stop": true
}
```

## Neuro API Bridge

This workspace vendors the public VedalAI Neuro SDK repository under
`vendor/neuro-sdk` for protocol reference. The local clone currently points at
commit `631314ab4b556c452c9380eaa14e7dc0074e5b31`; its license file is MIT and
the bridge reads the public API specification from
`vendor/neuro-sdk/API/SPECIFICATION.md`.

Hermes does not embed the Unity or Godot SDKs. It implements the public
websocket protocol shape in Python:

- client to Neuro: `startup`, `context`, `actions/register`, optional `actions/force`, and `action/result`;
- Neuro to client: `action`;
- `action.data` is parsed as untrusted JSON and validated locally.

The bridge surface is:

- `vrchat_neuro_status` - read-only vendor/profile/action catalog status.
- `vrchat_neuro_build_messages` - build bootstrap messages for a websocket harness.
- `vrchat_neuro_handle_action` - validate one incoming Neuro `action` and route it through the VRChat autonomy safety gate.
- `vrchat_autonomy_heartbeat_tick` - combine launch/readiness heartbeat with a profile-driven tick when readiness becomes actionable.
- `vrchat_autonomy_conversation_dry_run` - dry-run a multimodal conversation proof through local planning and Neuro routing.
- `vrchat_observation_ingest` - queue STT, vision, stream, operator, system, or textBox observations.
- `vrchat_observation_from_osc` - convert one incoming OSC event into a queued observation.
- `vrchat_observation_queue_status` - read-only queue count and preview.
- `vrchat_autonomy_preflight_bundle` - collect the read-only evidence bundle needed before a private live smoke.
- `vrchat_autonomy_wait_ready` - poll read-only readiness while the operator starts VRChat and VOICEVOX.
- `vrchat_autonomy_wait_then_tick` - wait for readiness, then run one gated heartbeat/profile tick.
- `vrchat_autonomy_runtime_doctor` - diagnose startup mismatches with read-only process, HTTP, port, profile, and audio evidence.
- `vrchat_autonomy_prepare_private_smoke` - evaluate live-smoke gates and build a dry-run action plan without live execution.
- `vrchat_autonomy_wait_then_private_smoke` - wait for readiness, then stop at private smoke preparation unless live smoke is explicitly allowed.
- `vrchat_autonomy_completion_audit` - read-only requirement audit for the full VRChat autonomy objective.
- `scripts/vrchat_neuro_bridge.py` - optional websocket harness for a local Neuro API server.
- `scripts/vrchat_observation_harness.py` - optional JSONL and OSC observation harness.
- `scripts/vrchat_heartbeat_tick.py` - optional heartbeat-to-profile-tick CLI for scheduler or manual runs.
- `scripts/vrchat_conversation_dry_run.py` - optional dry-run multimodal conversation proof.
- `scripts/vrchat_preflight.py` - optional read-only readiness/profile/audio-device evidence bundle.
- `scripts/vrchat_wait_ready.py` - optional read-only wait loop for VRChat/VOICEVOX startup readiness.
- `scripts/vrchat_wait_then_tick.py` - optional readiness wait followed by one gated profile tick.
- `scripts/vrchat_runtime_doctor.py` - optional read-only runtime mismatch doctor.
- `scripts/vrchat_wait_then_private_smoke.py` - optional readiness wait followed by private smoke preparation.
- `scripts/vrchat_completion_audit.py` - optional read-only completion audit for goal evidence.
- `skills/gaming/neuro-vrchat/SKILL.md` - Hermes skill guidance for this bridge.

The exposed Neuro action names are intentionally small:

- `vrchat_autonomy_turn`
- `vrchat_speak`
- `vrchat_chatbox`
- `vrchat_avatar_action` only when the profile contains allowed avatar actions

Incoming Neuro actions never grant raw OSC access. Live VRChat OSC or VOICEVOX
audio still requires a valid enabled profile, `dry_run: false`, and the explicit
live actuation acknowledgement string. Disabled, missing, invalid, or dry-run
profiles return a Neuro `action/result` without sending OSC or audio.

For a heartbeat automation, keep the first production wiring in dry-run mode.
Only switch `dry_run` to `false` after a private-instance manual smoke test has
confirmed VRChat OSC, VOICEVOX, virtual cable routing, and the avatar action
profile.

## Operator Profile

Use `vrchat_autonomy_prepare_profile` to create or refresh a local operator
profile, `vrchat_autonomy_profile_status` to validate it, and
`vrchat_autonomy_profile_tick` when a heartbeat automation should run one tick
from that profile.

Default path:

```text
<Hermes home>/config/vrchat-autonomy-profile.json
```

Example profile:

```text
docs/migration/vrchat_autonomy_profile.example.json
```

The example is intentionally disabled, observe-only, and dry-run. A profile
that sets `dry_run` to `false` is rejected unless it contains this exact
acknowledgement:

```text
I understand this sends OSC and/or audio to VRChat.
```

Prepare an enabled private-test dry-run profile for VOICEVOX and ChatBox via
the virtual cable output:

```powershell
py -3.12 scripts\vrchat_profile.py `
  --profile <Hermes home>\config\vrchat-autonomy-profile.json `
  --audio-output-device "CABLE Input" `
  --vrchat-microphone-device "CABLE Output"
```

The command preserves existing approved avatar actions unless replacement
actions are supplied through the tool API. It never writes a non-dry-run profile
unless `--arm-live` and the exact acknowledgement are both supplied.
The read-only preflight bundle also includes a `commands` block with the dry-run
profile preparation command, the acknowledgement printer, and the live-only
follow-up commands for private verification.

Approved avatar actions are profile-gated. Parameter writes must use plain
avatar parameter names, not raw OSC paths. Values must be bool, int, or float.
Optional `reset_after_sec` and `reset_value` can be used for pulse-style
parameters.

## Heartbeat-Triggered Profile Tick

Use `vrchat_autonomy_heartbeat_tick` when a scheduler or heartbeat should first
check launch/readiness state, then run one profile tick only when there is a
ready event. By default, a profile tick runs only on:

- `VRCHAT_LAUNCHED_READY`
- `READINESS_COMPLETE`

It returns `HEARTBEAT_NO_TICK` when VRChat is absent, readiness is incomplete,
or the heartbeat is already stable and no tick was requested.

The profile tick is still profile-gated. A live profile with `dry_run: false`
will not run from heartbeat unless the caller supplies both:

- `allow_live_profile: true`
- `live_ack: I understand this sends OSC and/or audio to VRChat.`

Manual dry-run command:

```powershell
py -3.12 scripts\vrchat_heartbeat_tick.py `
  --profile <Hermes home>\config\vrchat-autonomy-profile.json `
  --audio-output-device "CABLE Input" `
  --observation-json "{\"source\":\"operator\",\"text\":\"private readiness smoke\"}"
```

Scheduler-style runs should normally omit `--tick-when-already-ready` so the
loop does not repeatedly speak on a stable heartbeat. Use
`--tick-when-already-ready` only for deliberate dry-run testing, and use
`--force-tick` only for manual operator checks.

## Dry-Run Turn Planning

Use `vrchat_autonomy_plan_turn` to validate observations and a decision. It defaults to dry-run:

```json
{
  "observations": [
    {"source": "visionObservation", "summary": "The user is waving."}
  ],
  "decision": {
    "speak_text": "Thanks for waving.",
    "chatbox_text": "Thanks.",
    "emotion": "happy",
    "avatar_action": "wave"
  },
  "mode": "private_test",
  "allow_voice": true,
  "allow_chatbox": true,
  "dry_run": true,
  "output_device": "CABLE Input",
  "avatar_action_profiles": {
    "wave": [{"name": "Wave", "value": true}]
  }
}
```

Only set `dry_run` to `false` after:

- VRChat is running in a private or otherwise safe test instance.
- VOICEVOX is reachable.
- The virtual cable playback and VRChat microphone-side devices are confirmed.
- The avatar action profile has been reviewed.
- Emergency stop behavior is known.

## Private Smoke Gate

Run a read-only preflight bundle first:

```powershell
py -3.12 scripts\vrchat_preflight.py `
  --profile <Hermes home>\config\vrchat-autonomy-profile.json `
  --audio-output-device "CABLE Input" `
  --max-audio-devices 60 `
  --include-voicevox-synthesis
```

This checks VRChat, `python-osc`, VOICEVOX, the optional harness, the local
profile, the observation queue, the vendored Neuro SDK, output-capable audio
devices, the virtual cable playback/microphone-side route, and optional
VOICEVOX no-playback synthesis. It does not send OSC, play audio, record
microphone input, or open a Neuro websocket.

If the preflight does not match what the operator sees on screen, run the
runtime doctor with the operator flags:

```powershell
py -3.12 scripts\vrchat_runtime_doctor.py `
  --profile <Hermes home>\config\vrchat-autonomy-profile.json `
  --audio-output-device "CABLE Input" `
  --operator-reported-vrchat `
  --operator-reported-voicevox `
  --output _docs\2026-05-23_vrchat_runtime_doctor_Codex.json
```

The doctor is intended for the mismatch case. It can show whether VOICEVOX is
answering on another local URL, whether expected ports are visible, whether a
VRChat-like or VOICEVOX-like window exists under another visible process,
whether local launch candidates were found in bounded Steam/VOICEVOX locations,
whether relevant VRChat/VOICEVOX processes are visible to the current Windows
session,
and which readiness blocker should be fixed before attempting a live private
smoke.

Run the completion audit when deciding whether the full objective is actually
done:

```powershell
py -3.12 scripts\vrchat_completion_audit.py `
  --profile <Hermes home>\config\vrchat-autonomy-profile.json `
  --audio-output-device "CABLE Input" `
  --output _docs\2026-05-22_vrchat_completion_audit_Codex.json
```

The audit is read-only. It reports each requirement as `achieved` or
`incomplete`, performs one dry-run multimodal turn plan from synthetic
`visionObservation` and `operator` context, force-routes one synthetic Neuro
`vrchat_autonomy_turn` action through the Hermes safety gate in dry-run mode,
and keeps live verification incomplete until the current runtime is ready and a
private live smoke is deliberately armed and verified.

Use `vrchat_autonomy_prepare_private_smoke` or
`scripts/vrchat_private_smoke.py --prepare-only` immediately before live
attempts. It evaluates the live gate and builds the same representative
ChatBox, VOICEVOX, and avatar-action plan in dry-run mode, but it never sends
ChatBox, plays audio, or writes avatar parameters.

Use `vrchat_autonomy_private_smoke` or `scripts/vrchat_private_smoke.py` to
move from dry-run proof to a private-instance live smoke in small steps.

Use `vrchat_autonomy_wait_then_private_smoke` or
`scripts/vrchat_wait_then_private_smoke.py` when the operator has started
VRChat and VOICEVOX but readiness may still be settling. The harness first
polls the read-only readiness bundle. If readiness does not complete before
timeout, it writes a timeout result and performs no preparation or live action.
When readiness completes, it runs the private smoke preparation path. It still
does not send ChatBox, play VOICEVOX audio, or write avatar parameters unless
`--allow-live-smoke`, a non-dry-run armed profile, complete readiness, and the
exact live acknowledgement are all present.

The smoke gate always loads the local profile, checks readiness, and plans the
same ChatBox, VOICEVOX, and avatar action surfaces used by the autonomy loop.
It defaults to dry-run and will not execute live output unless all of these are
true:

- readiness is complete;
- the profile is valid and enabled;
- the profile has `dry_run: false`;
- the profile contains the exact live actuation acknowledgement;
- the command also supplies the exact live acknowledgement.

Print the required acknowledgement:

```powershell
py -3.12 scripts\vrchat_private_smoke.py --print-live-ack
```

Dry-run private smoke:

```powershell
py -3.12 scripts\vrchat_private_smoke.py `
  --profile <Hermes home>\config\vrchat-autonomy-profile.json `
  --audio-output-device "CABLE Input" `
  --avatar-action wave
```

Read-only live-smoke preparation:

```powershell
py -3.12 scripts\vrchat_private_smoke.py `
  --prepare-only `
  --profile <Hermes home>\config\vrchat-autonomy-profile.json `
  --audio-output-device "CABLE Input" `
  --avatar-action wave `
  --live-ack "I understand this sends OSC and/or audio to VRChat."
```

Wait for readiness, then prepare private smoke without live output:

```powershell
py -3.12 scripts\vrchat_wait_then_private_smoke.py `
  --profile <Hermes home>\config\vrchat-autonomy-profile.json `
  --audio-output-device "CABLE Input" `
  --avatar-action wave `
  --timeout-sec 120 `
  --interval-sec 5 `
  --live-ack "I understand this sends OSC and/or audio to VRChat."
```

Live private smoke, only after the operator has verified VRChat, VOICEVOX, and
the virtual cable manually:

```powershell
py -3.12 scripts\vrchat_private_smoke.py `
  --profile <Hermes home>\config\vrchat-autonomy-profile.json `
  --audio-output-device "CABLE Input" `
  --avatar-action wave `
  --live `
  --live-ack "I understand this sends OSC and/or audio to VRChat."
```

## Public Mode Defaults

For public instances:

- Keep movement disabled.
- Keep ChatBox disabled until deliberately enabled for a specific session.
- Keep speech short and rate limited.
- Prefer observe mode when unsure.

The current implementation builds the LLM request, validates/plans the
resulting decision, calls one configured Hermes auxiliary model for a single
safe turn, and provides a heartbeat/scheduler-friendly `vrchat_autonomy_loop_tick`
that consumes queued multimodal observations. It still does not start a hidden
always-on daemon by itself; activation is explicit through a scheduler,
heartbeat automation, CLI invocation, or another approved Hermes control path.

## Primary Source Grounding

Checked on 2026-05-22:

- [VRChat OSC Overview](https://docs.vrchat.com/docs/osc-overview): official OSC enablement, default ports `9000` and `9001`, and Python OSC library guidance.
- [VRChat OSC Avatar Parameters](https://docs.vrchat.com/docs/osc-avatar-parameters): official avatar parameter addresses and generated config behavior.
- [VRChat OSC as Input Controller](https://docs.vrchat.com/docs/osc-as-input-controller): official ChatBox limits and `/chatbox/input` argument shape.
- [VedalAI neuro-sdk API specification](https://github.com/VedalAI/neuro-sdk/blob/main/API/SPECIFICATION.md): websocket commands, action registration, action result behavior, and local validation responsibility.
- [VOICEVOX Engine repository](https://github.com/VOICEVOX/voicevox_engine): local HTTP engine used by the existing Hermes VOICEVOX integration.
