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

/**
 * Pure classifier for owner-typed messages that arrive in a GROUP chat.
 *
 * The DM gate below deliberately pre-filters group JIDs ("the gate doesn't
 * know about them"), so groups get their own decision: forwarding owner
 * messages out of a group is opt-in (bot mode + WHATSAPP_FORWARD_OWNER_MESSAGES)
 * and must pass the SAME group policy gate that inbound (non-fromMe) group
 * traffic uses — see `matchesInboundWhatsAppGroup` in `allowlist.js`.
 * Before this, fromMe group messages were dropped unconditionally, so an
 * allowlisted group could receive from Hermes but never talk back.
 *
 * Echoes of our own /send are NOT checked here; `bridge.js` drops them
 * downstream in the agent_echo check (recentlySentIds).
 *
 * Returned actions:
 *   - 'forward_owner' : opted in and the group passes the policy gate
 *   - 'drop_disabled' : not bot mode, or forwarding not opted into
 *   - 'drop_policy'   : opted in but the group fails the group policy gate
 */
export function classifyOwnerGroupMessageGate({
  botMode,
  forwardOwnerEnabled,
  groupPolicyMatches,
}) {
  if (!botMode || !forwardOwnerEnabled) {
    return { action: 'drop_disabled' };
  }
  if (!groupPolicyMatches) {
    return { action: 'drop_policy' };
  }
  return { action: 'forward_owner' };
}

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
