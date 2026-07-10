# MiniMax Voice Workflows Design

Date: 2026-07-10

## Goal

Make the Hermes Desktop Video Studio distinguish two MiniMax voice jobs that currently share one ambiguous form:

1. Use an existing MiniMax voice ID to generate and preview TTS audio.
2. Upload a voice sample to create and safely preview a new cloned voice ID.

The UI must not trigger the MiniMax ¥9.9 cloned-voice activation fee without a separate, explicit confirmation.

## Current Failure

The current "MiniMax 音色复刻" form always sends the entered ID and uploaded audio to `/voice_clone`. When a user enters a system voice such as `Korean_GentleBoss`, MiniMax correctly rejects the request because a clone ID must be new and unique.

The same action also sends `activate: true`. The sidecar interprets that by calling ordinary TTS after cloning. That ordinary TTS activates the new clone and can charge ¥9.9. The returned trial file is not rendered in the desktop UI, so the user can pay for activation without receiving an in-page preview.

## Considered Approaches

### A. Two explicit workflows with separate API actions — selected

Use two cards or tabs: "使用已有音色 ID" and "上传声音复刻". Each workflow owns its validation, request, result, preview player, and billing warning.

This makes the operation and cost visible before submission and prevents accidental routing to the clone endpoint.

### B. One form that auto-detects the operation

Infer "clone" when a file is attached and "existing ID" otherwise. This is compact but preserves ambiguity: stale file selections or accidental uploads can silently change the API action and cost.

Rejected because the billing boundary must be explicit.

### C. Keep the current form and only improve the duplicate-ID error

Detect provider/system IDs and tell the user to enter another ID. This removes the immediate error but still provides no existing-ID preview, no clone preview player, and no safe activation boundary.

Rejected because it treats the symptom rather than the workflow defect.

## Desktop Experience

### Workflow 1: Use an existing voice ID

The card contains:

- Voice ID input with provider-backed suggestions and manual entry.
- TTS model selector.
- Preview text.
- "生成试听" action.
- Audio player for the generated file.
- "选择用于视频" action after a successful preview.

Submission calls the existing MiniMax TTS route. It must not upload clone audio or call `/voice_clone`.

The selected video voice uses the existing MoneyPrinter encoding:

```text
minimax:<voice_id>
```

System voices only incur normal TTS character billing. Existing cloned voices may already be activated; the UI must not claim they are free.

### Workflow 2: Upload a voice sample and clone

The card contains:

- Required source audio, 10 seconds to 5 minutes, MP3/M4A/WAV, at most 20 MB.
- A generated unique Voice ID by default, with an advanced manual override.
- Optional prompt audio under 8 seconds plus the exact prompt transcript.
- Preview text.
- "创建克隆试听" action.
- Audio player for the clone endpoint's preview result.
- A separate "正式激活并选择" action with a clear ¥9.9 confirmation.

The clone preview action sends preview text through the MiniMax `/voice_clone` request itself and persists `demo_audio` locally for the desktop player. It must not call ordinary TTS and must leave local metadata as `activated: false`.

The activation action calls ordinary TTS only after explicit confirmation. On success it marks the metadata active and selects `minimax:<voice_id>` for video generation.

If the user does not activate within MiniMax's provider retention window, the preview clone may expire. The UI should state this without treating an unactivated preview as a durable voice.

## Voice Discovery

The existing local-only voice list is insufficient. The MoneyPrinter sidecar should query MiniMax `/v1/get_voice` for the current key and return categorized records:

- system voices;
- activated cloned voices;
- generated/designed voices;
- local unactivated clone previews.

The desktop can merge these records for suggestions while retaining category and activation state. A local metadata record from an older API key must not be presented as provider-confirmed access.

Manual ID entry remains supported.

## API Contracts

### Existing voice preview

Reuse:

```text
POST /api/capabilities/moneyprinter/minimax/tts
```

The desktop client adds a typed `generateMiniMaxTts` method and returns a media URL or file descriptor suitable for the authenticated desktop media protocol.

### Provider voice list

Extend:

```text
GET /api/capabilities/moneyprinter/minimax/voices
```

It returns structured voice records instead of only locally formatted strings. Backward compatibility is preserved where the MCP adapter expects local entries.

### Clone preview

Keep:

```text
POST /api/capabilities/moneyprinter/minimax/voices/clone
```

Change its preview behavior so `trial_text` is passed as MiniMax's inline clone `text` parameter. The response includes a locally persisted preview descriptor. `activate` defaults to false and the desktop clone-preview action never sets it true.

### Clone activation

Use ordinary TTS with the cloned ID after an explicit desktop confirmation. No automatic activation occurs inside clone creation.

## Validation and Errors

- Existing-ID preview requires a non-empty ID and preview text.
- Clone requires a new valid ID and valid source audio.
- A provider duplicate-ID response is translated into: "该 ID 已存在；请在已有音色中使用它，或为克隆生成新的 ID。"
- Prompt audio and prompt text are either both present or both absent.
- A failed preview never selects the voice.
- A successful clone preview does not label the voice activated.
- Paid activation shows the estimated one-time clone fee before submission.
- Long-running requests use endpoint-specific desktop timeouts rather than the 15-second default.

## Testing

Tests must cover:

1. Existing-ID preview routes to TTS and never to clone.
2. System ID `Korean_GentleBoss` can be submitted as TTS input.
3. Clone preview requires a new ID and source audio.
4. Clone preview sends inline preview text and does not call ordinary TTS.
5. Clone preview metadata remains unactivated.
6. Explicit activation calls TTS only after confirmation.
7. Duplicate-ID errors are translated into the actionable desktop message.
8. Provider and local voice records remain distinguishable.
9. Successful TTS and clone previews render playable audio.
10. Existing MoneyPrinter video generation still accepts `minimax:<voice_id>`.

Verification includes the focused Python tests, desktop Vitest suite, desktop typecheck, and a production desktop build. Live paid activation is excluded from automated verification.

## Out of Scope

- Deleting provider voices.
- Redesigning non-MiniMax TTS providers.
- Automatically purchasing voice activation.
- Running a live paid clone activation during tests.
