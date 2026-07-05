/**
 * Pure classifier for incoming non-fromMe WhatsApp messages.
 *
 * Self-chat mode normally rejects non-self messages so Hermes cannot become an
 * unsolicited WhatsApp bot. Human-in-the-loop/customer handoff deployments need
 * a narrow opt-in that lets customer DMs reach Python's pre_gateway_dispatch
 * plugin while preserving the safe default.
 */
export function classifyInboundMessageGate({
  fromMe,
  mode,
  allowNonSelf,
  allowlistMatches,
  senderId,
}) {
  if (fromMe) {
    return { action: 'pass' };
  }

  if (mode === 'self-chat') {
    return allowNonSelf
      ? { action: 'pass', reason: 'human_in_loop_allows_non_self' }
      : { action: 'drop', reason: 'self_chat_mode_rejects_non_self' };
  }

  if (!allowNonSelf && typeof allowlistMatches === 'function' && !allowlistMatches(senderId)) {
    return { action: 'drop', reason: 'allowlist_mismatch' };
  }

  if (allowNonSelf && typeof allowlistMatches === 'function' && !allowlistMatches(senderId)) {
    return { action: 'pass', reason: 'human_in_loop_bypasses_allowlist' };
  }

  return { action: 'pass' };
}
