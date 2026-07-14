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

export function registerReactionRoute(app, {
  getSocket,
  getConnectionState,
  sendWithTimeout,
}) {
  app.post('/react', async (req, res) => {
    if (!getSocket() || getConnectionState() !== 'connected') {
      return res.status(503).json({ error: 'Not connected' });
    }

    const { chatId, messageId, emoji, senderId, fromMe } = req.body;
    if (!chatId || !messageId || !emoji) {
      return res.status(400).json({ error: 'chatId, messageId, and emoji are required' });
    }

    try {
      const payload = buildReactionPayload({ chatId, messageId, emoji, senderId, fromMe });
      await sendWithTimeout(chatId, payload);
      return res.json({ success: true });
    } catch (err) {
      return res.status(500).json({ error: err.message });
    }
  });
}
