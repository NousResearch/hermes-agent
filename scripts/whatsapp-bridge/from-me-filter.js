/**
 * Pure helper for the bridge's fromMe (sent-by-self) message filter.
 *
 * Extracted from bridge.js so the group-vs-DM filtering rules are
 * unit-testable without booting Baileys / Express.
 *
 * @see scripts/whatsapp-bridge/from-me-filter.test.mjs
 * @see https://github.com/NousResearch/hermes-agent/issues/20143
 */

/**
 * Decide whether to drop a fromMe message at the bridge filter.
 *
 * Returns true when the message should be skipped (continue) and false when
 * it should be forwarded to the gateway.
 *
 * Echo-back protection on forwarded fromMe messages is still handled
 * downstream via REPLY_PREFIX startsWith() and recentlySentIds tracking
 * (bridge.js around line ~329).  This filter is intentionally narrow so
 * that group_policy / require_mention / group_allow_from rules in the
 * Python gateway can decide group-message routing.
 *
 * @param {object} options
 * @param {string} options.chatId - JID of the chat (e.g. ``120363001234@g.us``)
 * @param {boolean} options.isGroup - chatId.endsWith('@g.us')
 * @param {string} options.mode - ``'bot'`` or ``'self-chat'`` (WHATSAPP_MODE)
 * @param {string} [options.myNumber] - phone number portion of ``sock.user.id``
 * @param {string} [options.myLid] - lid portion of ``sock.user.lid``
 * @returns {boolean} true ⇒ drop the message at the bridge
 */
export function shouldFilterFromMeMessage({ chatId, isGroup, mode, myNumber, myLid }) {
  // Status broadcasts are always dropped — they're WhatsApp's broadcast
  // channel, never an interactive conversation.
  if (chatId.includes('status')) return true;

  // Bot mode: bridge runs as a separate WhatsApp number, so every fromMe
  // message is necessarily an echo of our own outgoing reply.
  if (mode === 'bot') return true;

  // Self-chat mode below this point.
  // Group messages get forwarded to the gateway so group_policy /
  // require_mention / group_allow_from can apply.  Echo-back protection
  // is already handled downstream via REPLY_PREFIX + recentlySentIds. (#20143)
  if (isGroup) return false;

  // DM: only the user's own self-chat is allowed.
  const chatNumber = chatId.replace(/@.*/, '');
  const isSelfChat =
    (myNumber && chatNumber === myNumber) ||
    (myLid && chatNumber === myLid);
  return !isSelfChat;
}
