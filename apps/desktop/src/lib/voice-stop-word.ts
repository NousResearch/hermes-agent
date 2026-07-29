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

// Canonical stop commands. Kept short and unambiguous; each must be the entire
// spoken utterance to match.
const STOP_PHRASES: readonly string[] = [
  'stop',
  'stop listening',
  'stop it',
  'stop please',
  'please stop',
  'stop stop',
  'that is all',
  "that's all",
  'never mind',
  'nevermind',
  'end conversation',
  'end the conversation',
  'goodbye',
  'good bye',
  'bye',
  'cancel'
]

// Optional address prefixes so "hermes stop" / "ok stop" / "hey hermes, stop"
// still count. Stripped before matching the core phrase.
const ADDRESS_PREFIXES: readonly string[] = ['hey hermes', 'hey hermes,', 'hermes', 'hermes,', 'ok', 'okay', 'hey']

// Normalise: lowercase, strip surrounding punctuation/whitespace, collapse
// internal runs of spaces. Trailing punctuation (".", "!", "…") is common in
// STT output and must not defeat the match.
function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[.,!?;:…]+/g, ' ')
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
export function isVoiceStopCommand(transcript: string): boolean {
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

  for (const candidate of candidates) {
    if (STOP_PHRASES.includes(candidate)) {
      return true
    }
  }

  return false
}
