export function buildReactionPayload({ chatId, messageId, emoji, senderId, fromMe = false }) {
  const key = {
    remoteJid: chatId,
    id: messageId,
    fromMe: !!fromMe,
  };
  if (String(chatId || '').endsWith('@g.us') && senderId) {
    key.participant = senderId;
  }

  return {
    react: {
      text: emoji,
      key,
    },
  };
}
