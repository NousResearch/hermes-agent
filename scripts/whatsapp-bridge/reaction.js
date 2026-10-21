import { reactionTarget } from './bridge_helpers.js';

export async function handleReactionRequest(req, res, { connected, messageStore, expandAliases, send, trackSent }) {
  if (!connected()) return res.status(503).json({ error: 'Not connected to WhatsApp' });
  const { chatId, messageId, emoji } = req.body || {};
  if (typeof chatId !== 'string' || !chatId || typeof messageId !== 'string' || !messageId
      || typeof emoji !== 'string' || emoji.length > 32) {
    return res.status(400).json({ error: 'chatId, messageId, and emoji are required' });
  }
  const target = reactionTarget(messageStore, chatId, messageId, expandAliases);
  if (!target) return res.status(404).json({ error: 'Message not found in this chat' });
  try {
    const sent = await send(target.chatId, { react: { text: emoji, key: target.key } });
    trackSent?.(sent);
    return res.json({ success: true });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
