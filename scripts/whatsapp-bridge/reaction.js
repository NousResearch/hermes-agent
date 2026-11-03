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

// Well inside the adapter's 5 s request timeout and far below the 60 s default
// send timeout, so a stalled reaction cannot hold the shared send queue (and the
// replies queued behind it).
export const REACTION_SEND_TIMEOUT_MS = 4000;

export function registerReactionRoute(app, {
  getSocket,
  getConnectionState,
  sendWithTimeout,
  trackSentMessageId,
  newMessageId,
}) {
  // Without it our own reaction comes back from Baileys as a fromMe upsert that
  // intake cannot tell from an owner-typed one, and it loops as a new turn.
  if (typeof trackSentMessageId !== 'function') {
    throw new TypeError('registerReactionRoute: trackSentMessageId is required');
  }
  // The id is chosen and remembered before the send, so a reaction that lands
  // after the caller's deadline still has its echo recognized.
  if (typeof newMessageId !== 'function') {
    throw new TypeError('registerReactionRoute: newMessageId is required');
  }
  app.post('/react', async (req, res) => {
    if (!getSocket() || getConnectionState() !== 'connected') {
      return res.status(503).json({ error: 'Not connected' });
    }

    const { chatId, messageId, emoji, senderId, fromMe } = req.body;
    // Baileys uses an empty reaction text to remove the account's existing
    // reaction from the target message; reject only a missing field.
    if (!chatId || !messageId || emoji === undefined || emoji === null) {
      return res.status(400).json({ error: 'chatId, messageId, and emoji are required' });
    }

    try {
      const payload = buildReactionPayload({ chatId, messageId, emoji, senderId, fromMe });
      const reactionId = newMessageId();
      trackSentMessageId({ key: { id: reactionId } });
      await sendWithTimeout(chatId, payload, { messageId: reactionId }, REACTION_SEND_TIMEOUT_MS, { skipIfExpired: true });
      return res.json({ success: true });
    } catch (err) {
      return res.status(500).json({ error: err.message });
    }
  });
}
