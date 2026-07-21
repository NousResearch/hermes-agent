/**
 * Decide whether the DM sender allowlist should run for this inbound.
 *
 * Group chats are gated by WHATSAPP_GROUP_POLICY / WHATSAPP_GROUP_ALLOWED_USERS
 * on the Python adapter. Applying WHATSAPP_ALLOWED_USERS (DM sender allowlist)
 * to group participants blocked every non-owner member and made allowlisted
 * support groups look "dead".
 *
 * @param {object} opts
 * @param {boolean} opts.isGroup
 * @param {boolean} opts.fromMe
 * @param {string}  opts.dmPolicy   WHATSAPP_DM_POLICY
 * @returns {boolean} true if the bridge should enforce the DM sender allowlist
 */
export function shouldEnforceDmSenderAllowlist({
  isGroup = false,
  fromMe = false,
  dmPolicy = 'allowlist',
} = {}) {
  if (fromMe) return false;
  if (isGroup) return false;
  if (dmPolicy === 'pairing') return false;
  return true;
}
