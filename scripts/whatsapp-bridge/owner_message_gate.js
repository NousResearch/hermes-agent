/**
 * Pure classifier for the WhatsApp bridge's bot-mode dispatch loop.
 *
 * Centralises the "should this fromMe message be forwarded as fromOwner?"
 * decision so the gate can be unit-tested without spinning up Baileys or
 * the Express server.
 *
 * Lives next to `outbound_ids.js` rather than inline in `bridge.js`
 * because the previous implementation accidentally bypassed the
 * customer-side allowlist when forwarding owner-typed messages — see
 * the regression test in `owner_message_gate.test.mjs`.
 *
 * Caller responsibilities:
 *   - Only invoke in bot mode. Self-chat mode has its own self-chat
 *     pinning logic and must not delegate here.
 *   - Pre-filter group / status JIDs (the gate doesn't know about them).
 *   - On `drop_allowlist`, log the rejection so operators can audit
 *     accidental allowlist mismatches.
 *
 * Returned actions:
 *   - 'pass'           : non-fromMe, fall through to existing handling
 *   - 'drop_echo'      : fromMe and matches a recently-sent /send id
 *   - 'drop_disabled'  : fromMe but operator hasn't opted into forwarding
 *   - 'drop_allowlist' : fromMe and the *customer chatId* isn't on the
 *                        allowlist (owner-typed reply to a stranger)
 *   - 'forward_owner'  : fromMe, owner-typed, allowlisted — forward with
 *                        fromOwner: true
 */

export function classifyOwnerMessageGate({
  fromMe,
  fromOwnerEnabled,
  recentlySent,
  allowlistMatches,
  messageId,
  chatId,
}) {
  if (!fromMe) {
    return { action: 'pass' };
  }
  if (recentlySent && recentlySent.has(messageId)) {
    return { action: 'drop_echo' };
  }
  if (!fromOwnerEnabled) {
    return { action: 'drop_disabled' };
  }
  // Allowlist gate: check the *customer* chatId, not the sender. The
  // sender is the owner's own number/LID and won't be on the allowlist
  // by construction. Without this check, any contact the owner happens
  // to reply to leaks into Hermes and triggers implicit handover in the
  // gateway-policy plugin.
  if (typeof allowlistMatches === 'function' && !allowlistMatches(chatId)) {
    return { action: 'drop_allowlist' };
  }
  return { action: 'forward_owner' };
}

/**
 * Classify a fromMe group message before it can enter the inbound queue.
 *
 * Same-account group processing is intentionally separate from the owner-DM
 * path above: group authorization and mention handling belong to the Python
 * adapter, while this bridge-level gate only enforces the explicit opt-in and
 * suppresses echoes of messages sent through the bridge itself.
 */
export function classifyFromMeGroupGate({
  mode,
  enabled,
  recentlySent,
  messageId,
}) {
  if (mode !== 'bot') {
    return { action: 'drop_mode' };
  }
  if (recentlySent && recentlySent.has(messageId)) {
    return { action: 'drop_echo' };
  }
  if (!enabled) {
    return { action: 'drop_disabled' };
  }
  return { action: 'forward_group' };
}
