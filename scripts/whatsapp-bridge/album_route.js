import { prepareAlbumItems, sendAlbumSequence } from './bridge_helpers.js';

export async function sendAlbumMessageWithWatchdog({
  socket,
  chatId,
  payload,
  mediaUploadTimeoutMs,
  watchdogTimeoutMs,
  onFatalTimeout,
}) {
  let timer;
  const sendPromise = Promise.resolve().then(() => socket.sendMessage(
    chatId,
    payload,
    { mediaUploadTimeoutMs },
  ));
  const timeoutPromise = new Promise((_, reject) => {
    timer = setTimeout(() => {
      const error = new Error(`album send timed out after ${watchdogTimeoutMs}ms`);
      try {
        onFatalTimeout(error);
      } finally {
        // Production's fatal callback terminates the bridge synchronously, so
        // an unresolved Baileys send can never escape this queue slot. Tests
        // use a non-fatal callback and observe the rejection instead.
        reject(error);
      }
    }, watchdogTimeoutMs);
  });
  return Promise.race([sendPromise, timeoutPromise]).finally(() => clearTimeout(timer));
}

export function registerAlbumRoute(app, {
  getSocket,
  getConnectionState,
  enqueueSend,
  trackSentMessageId,
  messageStore,
  sendTimeoutMs = 60000,
  onFatalTimeout = () => {},
}) {
  app.post('/send-album', async (req, res) => {
    const socket = getSocket();
    if (!socket || getConnectionState() !== 'connected') {
      return res.status(503).json({
        success: false,
        attempted: false,
        status: 'not_connected',
        error: 'Not connected to WhatsApp',
      });
    }

    const { chatId, items } = req.body;
    if (!chatId || !Array.isArray(items)) {
      return res.status(400).json({
        success: false,
        attempted: false,
        status: 'validation_error',
        error: 'chatId and items are required',
      });
    }

    let preparedItems;
    try {
      preparedItems = prepareAlbumItems(items);
    } catch (error) {
      return res.status(400).json({
        success: false,
        attempted: false,
        status: 'validation_error',
        error: error.message,
      });
    }

    const result = await enqueueSend(() => sendAlbumSequence({
      chatId,
      items: preparedItems,
      // Do not race album socket writes against a timer. A losing sendMessage
      // promise cannot be cancelled safely. Baileys gets a real upload timeout
      // that destroys the HTTP request; a slightly longer watchdog terminates
      // the bridge if sendMessage itself still fails to settle.
      send: async (targetChatId, payload) => {
        const sent = await sendAlbumMessageWithWatchdog({
          socket,
          chatId: targetChatId,
          payload,
          mediaUploadTimeoutMs: Math.max(1000, sendTimeoutMs - 5000),
          watchdogTimeoutMs: sendTimeoutMs,
          onFatalTimeout,
        });
        trackSentMessageId(sent);
        messageStore.remember(sent);
        return sent;
      },
    }));

    const statusCode = result.success ? 200 : (result.status === 'partial_failure' ? 207 : 502);
    return res.status(statusCode).json(result);
  });
}
