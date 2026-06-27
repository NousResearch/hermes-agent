// Pure helpers for WhatsApp self-chat gating decisions.
//
// WhatsApp's multi-device stack now addresses accounts with a LID (Linked
// Identity Device) identifier — e.g. `158939576553626@lid` — alongside the
// classic phone-number JID — e.g. `923344176856@s.whatsapp.net`. As a result,
// a user's own "Message Yourself" messages can be delivered to a linked device
// with `key.fromMe === false`. The bridge must therefore recognise the user's
// own self-chat from the chat identifier itself, independently of `fromMe`,
// otherwise legitimate self-chat messages are dropped as "non-self".
//
// Kept as a side-effect-free module (mirrors allowlist.js) so the decision is
// unit-testable without booting a Baileys socket.

/**
 * Derive the account's own phone-number and LID identifiers from a Baileys
 * `sock.user` object, stripping the device suffix (`:NN`) and JID host.
 *
 * @param {{ id?: string, lid?: string } | null | undefined} user
 * @returns {{ myNumber: string, myLid: string }}
 */
export function deriveSelfIdentifiers(user) {
  const strip = (value) => (value || '').replace(/:.*@/, '@').replace(/@.*/, '');
  return { myNumber: strip(user?.id), myLid: strip(user?.lid) };
}

/**
 * Decide whether `chatId` is the account's own self-chat (the "Message
 * Yourself" conversation). Groups and status broadcasts are never self-chats.
 * Intentionally `fromMe`-agnostic: LID addressing can flip that flag, so the
 * decision relies solely on the chat identifier matching our own number/LID.
 *
 * @param {string} chatId Baileys remoteJid, e.g. `923344176856@s.whatsapp.net`.
 * @param {{ myNumber?: string, myLid?: string }} self From deriveSelfIdentifiers.
 * @returns {boolean}
 */
export function isOwnSelfChat(chatId, { myNumber, myLid } = {}) {
  if (!chatId) return false;
  if (chatId.endsWith('@g.us') || chatId.includes('status')) return false;
  const chatNumber = chatId.replace(/@.*/, '');
  return Boolean((myNumber && chatNumber === myNumber) || (myLid && chatNumber === myLid));
}
