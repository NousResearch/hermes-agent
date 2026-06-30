# Life Recording Shortcut MVP

Goal: phone-side "Meeting Capture" for long DJI Mic sessions: events, walks, investor meetings, dinners, and braindumps.

## Hardware

```text
DJI Mic transmitter on Samuel
→ wireless to DJI receiver
→ receiver plugged into iPhone USB-C
→ iPhone records audio
```

## Shortcut v0: safe event recorder

Name: `Start Life Capture`

This should be boring and reliable. For now it should **not** try to continuously stream audio in the background.

### Actions

1. **Ask for Input**
   - Prompt: `What are you recording?`
   - Default: `Life Capture`
   - Store as `Session Name`

2. **Current Date**
   - Format: `yyyy-MM-dd HH.mm`
   - Store as `Timestamp`

3. **Show Alert**
   - Title: `DJI check`
   - Message:
     ```text
     Receiver plugged into USB-C?
     TX clipped on / battery OK?
     Waveform moves if you tap the DJI mic?
     For important meetings, also start fallback Voice Memos if unsure.
     ```

4. **Record Audio**
   - Audio Quality: High
   - Start Recording: Immediately
   - Finish Recording: On Tap

5. **Set Name**
   - Name: `Life Capture - [Timestamp] - [Session Name].m4a`

6. **Save File**
   - Path: `iCloud Drive/Shortcuts/Life Captures/`
   - Ask Where to Save: Off
   - Overwrite If File Exists: Off

7. **Share / Send later**
   - MVP: show Share Sheet so Samuel can send to Hermes/Telegram/Drive manually.
   - Next: POST file to Hermes upload endpoint once implemented.

8. **Show Notification**
   - `Saved Life Capture - [Session Name]`

## Shortcut v1: upload to Hermes

Once the server endpoint exists:

```text
Record Audio
→ Save File locally
→ Get Contents of URL
   POST https://<hermes-host>/api/mobile/capture-segment
   multipart/form-data:
     audio: file
     source: iphone-shortcut
     sessionName: user input
     startedAt: timestamp
     device: iPhone
→ Show response summary or queued/uploaded status
```

## Shortcut v1.5: transcript ingress

If using Hermes `/api/audio/transcribe` + `/api/voice-ingest` instead of file upload:

```text
Record Audio
→ Base64 Encode file
→ POST /api/audio/transcribe with data URL
→ Extract transcript
→ POST /api/voice-ingest JSON:
   {
     "text": "Process this life capture transcript: ...",
     "source": "iphone-life-capture",
     "timestamp": ISO timestamp,
     "user_id": "samuel"
   }
```

This is less good for very long sessions because Shortcuts may struggle with huge base64 payloads.

## Reliability rules

- Use the native recording action / recorder app as source of truth. Do not depend on Hermes being online during the event.
- Always save locally first, upload second.
- Keep the long original recording/transcript as the canonical artifact. Long transcripts are gold: they preserve context, exact phrasing, sequence, and later-discovered details.
- Do **not** ask Samuel to manually stop every 60–90 minutes as the default. That is only an emergency fallback if a specific recording app proves it cannot safely finalize multi-hour files.
- For critical meetings, run a 30-second test recording first and play it back.
- If iOS kills a background upload, that is fine; retry later from the saved file.

## Long-session artifact policy

The app should distinguish **canonical artifacts** from **processing chunks**:

1. Canonical original audio
   - One continuous recording per session whenever possible.
   - Never delete automatically.
   - Stored locally first, then uploaded/backed up.

2. Canonical full transcript
   - One full transcript stitched in chronological order.
   - Keep timestamps and markers.
   - This is the source of truth for later search, summaries, and memory extraction.

3. Processing chunks
   - Internal only: split audio/transcript for STT/model limits, retries, and parallel processing.
   - Chunk boundaries must not become the user-facing artifact.
   - The summarizer/extractor merges chunks back into a single event/session view.

## Hermes app target

Hermes should absorb this shortcut into the app:

- one-tap `Start Life Capture`
- external mic input label
- timer + meter
- important marker
- continuous original recording preserved as canonical
- internal rolling/chunked processing for upload/STT only
- local queue
- upload/retry
- full canonical transcript + searchable event page
- Hermes summary/action extraction
- approval cards for calendar/memory/follow-ups
