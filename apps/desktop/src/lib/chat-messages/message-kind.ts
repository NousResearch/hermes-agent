// Message-kind classification for user-role rows on the desktop.
//
// A union-typed session stores three kinds of rows on the `user` role:
//   1. Inter-agent deliveries  ("Message from 🤖 <sender>: …")
//   2. Injected system notices ("[IMPORTANT: Background process …]")
//   3. Real human prompts      (everything else)
//
// All three render differently (attributed agent card, compact system strip,
// normal user bubble) and the inter-agent collapse gate must treat only the
// third as evidence a human is in the thread. Keeping the matchers in one
// module lets the runtime converter stamp an authoritative `isHuman` flag and
// stops the render layer from re-deriving kind from raw text.

// Agent-to-agent deliveries ("Message from 🤖 <sender>: …", the Bot Mode /
// multi-profile convention; optional "(@<handle>)" carries the sender's
// profile name for avatar resolution; legacy "[Message from agent
// '<sender>'] …" too). They arrive on the user role because the recipient's
// turn runs on it, but they are NOT the human speaking.
export const AGENT_MESSAGE_RE =
  /^(?:Message from (?:🤖\s*)?([^:\n(]{1,64}?)(?:\s*\(@([a-z0-9][a-z0-9_-]{0,63})\))?:\s*|\[Message from agent '([^']{1,64})'\]\s*)([\s\S]*)$/u

// Injected as user messages for alternation; not human prompts (thread.tsx).
export const PROCESS_NOTIFICATION_RE = /^\[IMPORTANT: Background process [\s\S]*\]$/

/** True when a trimmed user-role text is a genuine human prompt (not an
 *  inter-agent delivery and not an injected background-process notice). */
export function isHumanUserText(text: string): boolean {
  const trimmed = text.trim()
  return !AGENT_MESSAGE_RE.test(trimmed) && !PROCESS_NOTIFICATION_RE.test(trimmed)
}