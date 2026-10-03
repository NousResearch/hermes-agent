/**
 * Self-chat identity and the outgoing reply prefix.
 *
 * In self-chat mode the agent and the user share one WhatsApp number, so
 * replies in the self-chat thread carry a prefix to tell them apart from
 * what the user typed. That ambiguity only exists in that one thread: a
 * message the agent sends to anyone else (cron delivery, `hermes send`,
 * send_message to a contact) is unambiguous, and prefixing it would tell
 * the recipient it came from automation.
 *
 * Pure so it can be unit-tested without Baileys or the Express server.
 */

// "447700900001:12@s.whatsapp.net" / "123456789:12@lid" -> "447700900001" / "123456789"
function accountNumber(jid) {
  return String(jid || '').replace(/:.*@/, '@').replace(/@.*/, '');
}

/**
 * Whether `chatId` is the linked account's own chat. `user` is Baileys'
 * `sock.user` ({ id, lid }); WhatsApp addresses the self-chat by either
 * the phone-number JID or the LID.
 */
export function isSelfChatId(chatId, user) {
  const chatNumber = String(chatId || '').replace(/@.*/, '');
  if (!chatNumber) return false;
  const myNumber = accountNumber(user?.id);
  const myLid = accountNumber(user?.lid);
  return Boolean((myNumber && chatNumber === myNumber) || (myLid && chatNumber === myLid));
}

/**
 * Prepend `replyPrefix` only for self-chat mode AND a send to the self-chat.
 * Bot mode never needs it: the bot has its own number.
 */
export function formatOutgoingMessage(message, { mode, replyPrefix, chatId, user }) {
  if (mode !== 'self-chat' || !replyPrefix) return message;
  if (!isSelfChatId(chatId, user)) return message;
  return `${replyPrefix}${message}`;
}
