// Spoken stop-word detection for the voice conversation loop.
//
// When someone is in a hands-free "Hey Hermes" voice chat, the natural way to
// end it is to SAY "stop" — not reach for the mouse. Without this, a spoken
// "stop" is just transcribed and sent to the agent as a normal turn, so the
// conversation never ends (the reported bug). This matcher recognises a short
// utterance whose entire content is a stop command and ends the conversation
// instead of submitting it.
//
// Deliberately conservative: it only fires when the WHOLE utterance is a stop
// phrase (optionally addressed to Hermes), so a real turn that merely contains
// the word "stop" — e.g. "stop the docker container" or "how do I stop a
// running process" — is never swallowed.

// Match the backend's voice.stop_phrases default. Explicit lists replace it;
// an empty list disables interception rather than restoring a hidden default.
export const DEFAULT_VOICE_STOP_PHRASES: readonly string[] = ['stop']

// Optional address prefixes so "hermes stop" / "ok stop" / "hey hermes, stop"
// still count. Stripped before matching the core phrase.
const ADDRESS_PREFIXES: readonly string[] = ['hey hermes', 'hey hermes,', 'hermes', 'hermes,', 'ok', 'okay', 'hey']

// Normalise: lowercase, strip surrounding punctuation/whitespace, collapse
// internal runs of spaces. Trailing punctuation (".", "!", "…") is common in
// STT output and must not defeat the match.
function normalize(text: string): string {
  return text
    .normalize('NFC')
    .toLowerCase()
    .replace(/[.,!?;:…]+/g, ' ')
    .replace(/^[\s"'“”‘’]+|[\s"'“”‘’]+$/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function stripAddress(text: string): string {
  for (const prefix of ADDRESS_PREFIXES) {
    if (text === prefix) {
      // Bare address ("hermes") is not a stop command on its own.
      continue
    }

    if (text.startsWith(`${prefix} `)) {
      return text.slice(prefix.length + 1).trim()
    }
  }

  return text
}

/**
 * True when the entire spoken utterance is a stop command (optionally addressed
 * to Hermes). Returns false for anything that merely contains "stop" as part of
 * a longer, substantive request.
 */
export function isVoiceStopCommand(
  transcript: string,
  stopPhrases: readonly string[] = DEFAULT_VOICE_STOP_PHRASES
): boolean {
  if (!transcript) {
    return false
  }

  const normalized = normalize(transcript)

  if (!normalized) {
    return false
  }

  // Match with the address prefix stripped, and also as-is (so a bare "stop"
  // with no prefix still matches, and "please stop" — where "please" isn't a
  // prefix — matches directly).
  const candidates = new Set([normalized, stripAddress(normalized)])

  for (const phrase of stopPhrases) {
    if (candidates.has(normalize(phrase))) {
      return true
    }
  }

  return false
}

/**
 * Typed-stop interception decision for the composer: a bare stop command
 * typed while the voice conversation is live ends the conversation instead of
 * being sent as a turn. Attachments mean the message is a real payload —
 * never intercepted. Outside a voice conversation typed text always passes
 * through unchanged.
 */
export function interceptsTypedVoiceStop(
  conversationActive: boolean,
  text: string,
  attachmentCount = 0,
  stopPhrases: readonly string[] = DEFAULT_VOICE_STOP_PHRASES
): boolean {
  return conversationActive && attachmentCount === 0 && isVoiceStopCommand(text, stopPhrases)
}
