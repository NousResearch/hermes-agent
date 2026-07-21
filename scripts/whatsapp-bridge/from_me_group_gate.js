/**
 * Pure classifier for fromMe messages that land in a WhatsApp group.
 *
 * Stock Hermes dropped every fromMe group message (`from_me_group`), which
 * blocked two legitimate paths:
 *   1. self-chat mode operators talking to the agent in a group
 *   2. bot-mode personal-number deployments where the linked phone is the
 *      owner's own number and they type in an allowlisted group
 *
 * Group membership / group_policy is enforced on the Python side
 * (WHATSAPP_GROUP_ALLOWED_USERS). This gate only decides whether the
 * bridge should forward a fromMe group message at all.
 *
 * @param {object} opts
 * @param {boolean} opts.isGroup
 * @param {boolean} opts.fromMe
 * @param {boolean} opts.isStatus
 * @param {string}  opts.mode                 'bot' | 'self-chat'
 * @param {boolean} opts.forwardOwnerMessages WHATSAPP_FORWARD_OWNER_MESSAGES
 * @param {boolean} opts.recentlySent         true if this message id was our /send
 * @returns {{ action: 'not_applicable'|'drop_status'|'drop_echo'|'drop_from_me_group'|'forward_owner' }}
 */
export function classifyFromMeGroupGate({
  isGroup = false,
  fromMe = false,
  isStatus = false,
  mode = 'bot',
  forwardOwnerMessages = false,
  recentlySent = false,
} = {}) {
  if (!fromMe) {
    return { action: 'not_applicable' };
  }
  if (isStatus) {
    return { action: 'drop_status' };
  }
  if (!isGroup) {
    return { action: 'not_applicable' };
  }
  if (recentlySent) {
    return { action: 'drop_echo' };
  }
  // Self-chat: the linked account is the only speaker the agent ever sees.
  // Group fromMe must reach Python so group_policy / group_allow_from apply.
  if (mode === 'self-chat') {
    return { action: 'forward_owner' };
  }
  // Bot mode: only when the operator opts into owner-typed forwards
  // (personal-number bot testing). Default stays drop_from_me_group so
  // dedicated bot-number deployments are unchanged.
  if (forwardOwnerMessages) {
    return { action: 'forward_owner' };
  }
  return { action: 'drop_from_me_group' };
}
