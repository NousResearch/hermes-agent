/**
 * Detect the per-session exclusivity refusal that arrives as a Desktop turn
 * error (gateway JSON-RPC 4090 → inline assistant error bubble).
 *
 * The lease/ownership guard itself is correct; Desktop must still offer an
 * escape (start a new session) because Retry only reproduces the same refusal.
 * Match the stable English contract from
 * `hermes_cli.active_sessions.session_already_owned_message` plus the machine
 * reason code when present in diagnostics text.
 */
export function isSessionOwnershipRefusal(text: string | null | undefined): boolean {
  const value = (text || '').trim()
  if (!value) {
    return false
  }

  if (/\balready has a live owner\b/i.test(value)) {
    return true
  }

  // Machine reason may appear in copied diagnostics or structured error dumps.
  return /\bSESSION_NOT_OWNED\b/.test(value)
}
