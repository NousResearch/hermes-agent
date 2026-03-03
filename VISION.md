# Hermes Agent — Vision Board & Roadmap

A living brainstorming doc for features, ideas, and strategic direction.
Last updated: March 2, 2026

---

## Voice Mode

**Inspiration:** Claude Code's /voice rollout (March 2026) — lets users talk
to the coding agent instead of typing, toggled with a slash command.

### CLI UX (primary target)

The voice mode lives inside the existing CLI terminal experience:

1. **Activation:** User types `/voice` in the Hermes CLI to toggle voice on/off
2. **Status indicator:** A persistent banner appears at the top of the prompt
   area: `Voice mode enabled — hold Space to speak`
3. **Push-to-talk:** User holds the Space bar to record. Releasing sends the
   audio for transcription. The input prompt placeholder changes to guide:
   `> hold space bar to speak`
4. **Transcription:** Speech is transcribed to text and submitted as a normal
   user message — the agent processes it identically to typed input
5. **Agent response:** Text response streams to the terminal as usual.
   Optionally, TTS can read the response aloud (we already have
   text_to_speech). Could be a `/voice tts` sub-toggle.
6. **Deactivation:** `/voice` again to toggle off, returns to normal typing

**Implementation notes:**
- Push-to-talk needs raw terminal/keyboard input (prompt_toolkit has key
  binding support — we already use it for the CLI input)
- Audio capture via PyAudio or sounddevice, stream to STT provider
- Visual feedback while recording: waveform animation or pulsing indicator
  in the terminal (could use rich/textual for this)
- Space bar hold must NOT conflict with normal typing when voice is off

### Gateway Platforms

- **Telegram:** Already receives voice messages natively — transcribe them
  automatically with STT and process as text. Users already send voice
  notes; we just need to handle the audio file.
- **Discord:** Similar — voice messages come as attachments, transcribe and
  process
- **WhatsApp:** Voice notes are a primary interaction mode, same approach

### Ideas

- Agent can already do TTS output (text_to_speech tool exists) — pair with
  voice input for a full conversational loop
- Latency matters — voice conversations feel bad above ~2s response time
- Could adjust system prompt in voice mode to be more concise/conversational
- Audio cues for tool call confirmations, errors, completion
- Streaming STT (transcribe while user is still speaking) for lower latency

### Open Questions

- Which STT provider? (Whisper local, Deepgram, AssemblyAI, etc.)
  - Local Whisper = no API dependency but needs GPU for speed
  - Deepgram/AssemblyAI = fast streaming, but adds a service dependency
- Should voice mode change the system prompt to be more conversational/concise?
- How to handle tool call confirmations in voice — audio cues?
- Do we want full duplex (agent can interrupt/be interrupted) or half-duplex?

---

## Ideas Backlog

*(New ideas get added here, then organized into sections as they mature)*

---

## Shipped

*(Track completed vision items here for posterity)*
