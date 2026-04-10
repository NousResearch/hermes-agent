/**
 * session-bridge.mjs
 *
 * Node.js HTTP bridge server wrapping @bonesgit/session-desktop-library SessionClient.
 * Exposes the SessionClient API over HTTP and SSE for a Python gateway adapter.
 *
 * Usage:
 *   Normal:  node session-bridge.mjs
 *   Setup:   node session-bridge.mjs --setup
 *
 * Environment variables:
 *   SESSION_MNEMONIC    - 13-word space-separated mnemonic (required for normal mode if not registered)
 *   SESSION_DATA_PATH   - path to session data directory (required)
 *   SESSION_BOT_NAME    - display name for the bot (default: 'Hermes')
 *   SESSION_BRIDGE_PORT - HTTP port (default: 8095)
 *   SESSION_LOG_LEVEL   - log level passed to SessionClient (default: 'warn')
 */

import fs from 'fs';
import http from 'http';
import { SessionClient } from '@bonesgit/session-desktop-library';

// Log library version immediately on load
try {
  const pkgPath = new URL('./node_modules/@bonesgit/session-desktop-library/package.json', import.meta.url);
  const { version } = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
  console.log(`[session-bridge] session-desktop-library v${version}`);
} catch {
  console.log('[session-bridge] session-desktop-library version unknown');
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const DATA_PATH  = process.env.SESSION_DATA_PATH;
const BOT_NAME   = process.env.SESSION_BOT_NAME   || 'Hermes';
const PORT       = parseInt(process.env.SESSION_BRIDGE_PORT || '8095', 10);
const LOG_LEVEL  = process.env.SESSION_LOG_LEVEL  || 'warn';
// NOTE: SESSION_MNEMONIC is only used in --setup mode, NEVER logged
const MNEMONIC   = process.env.SESSION_MNEMONIC;

if (!DATA_PATH) {
  console.error('ERROR: SESSION_DATA_PATH is required.');
  process.exit(1);
}

// ---------------------------------------------------------------------------
// --check mode: report whether an account is registered, never create one
// ---------------------------------------------------------------------------
if (process.argv.includes('--check')) {
  const checkClient = new SessionClient({ dataPath: DATA_PATH, logLevel: LOG_LEVEL });
  await checkClient.initialize();
  if (checkClient.isRegistered()) {
    console.log(JSON.stringify({ status: 'existing', sessionId: checkClient.getSessionId() }));
  } else {
    console.log(JSON.stringify({ status: 'not_registered' }));
  }
  await checkClient.shutdown();
  process.exit(0);
}

// ---------------------------------------------------------------------------
// --setup mode: bootstrap account and print JSON, then exit
// ---------------------------------------------------------------------------
if (process.argv.includes('--setup')) {
  await runSetup();
  process.exit(0);
}


let isReady = false;
/** @type {http.ServerResponse|null} */
let currentSseClient = null;
/** @type {http.ServerResponse|null} */
let currentConvSseClient = null;

const client = new SessionClient({ dataPath: DATA_PATH, logLevel: LOG_LEVEL });

// Bootstrap: initialize, register/restore account, start HTTP server
await bootstrap();

// ---------------------------------------------------------------------------
// Setup function
// ---------------------------------------------------------------------------
async function runSetup() {
  if (!DATA_PATH) {
    console.error('ERROR: SESSION_DATA_PATH is required for --setup mode.');
    process.exit(1);
  }

  const setupClient = new SessionClient({ dataPath: DATA_PATH, logLevel: LOG_LEVEL });
  await setupClient.initialize();

  if (setupClient.isRegistered()) {
    const sessionId = setupClient.getSessionId();
    console.log(JSON.stringify({ status: 'existing', sessionId }));
  } else if (MNEMONIC) {
    // Restore an existing account from a known mnemonic
    const sessionId = await setupClient.restoreAccount(MNEMONIC.trim());
    console.log(JSON.stringify({ status: 'restored', sessionId }));
  } else {
    // Generate a fresh mnemonic and create a new account
    const mnemonic = await SessionClient.generateMnemonic();
    const sessionId = await setupClient.createAccount(mnemonic, BOT_NAME);
    // Output mnemonic ONLY to stdout inside JSON — never to stderr
    console.log(JSON.stringify({ status: 'created', sessionId, mnemonic }));
  }

  await setupClient.shutdown();
}

// ---------------------------------------------------------------------------
// Bootstrap (normal mode)
// ---------------------------------------------------------------------------
async function bootstrap() {
  await client.initialize();

  if (client.isRegistered()) {
    console.log(`[session-bridge] Account registered: ${client.getSessionId()}`);
  } else {
    console.error('ERROR: No account registered in data directory. ' +
      'Run `hermes setup` to configure Session first.');
    process.exit(1);
  }

  // Subscribe to incoming messages and forward to SSE clients
  subscribeToMessages();
  subscribeToConversations();

  // Start HTTP server
  const server = http.createServer(requestHandler);
  server.listen(PORT, () => {
    console.error(`[session-bridge] Listening on port ${PORT}, sessionId=${client.getSessionId()}`);
  });

  // Mark as ready and notify any waiting SSE client
  isReady = true;
  sendReadyEvent();
}

// ---------------------------------------------------------------------------
// Message subscription
// ---------------------------------------------------------------------------
function subscribeToMessages() {
  // client.messages() returns an AsyncIterable — iterate it in the background
  (async () => {
    try {
      for await (const msg of client.messages()) {
        if (currentSseClient) {
          try {
            currentSseClient.write(`data: ${JSON.stringify({ type: 'message', data: msg })}\n\n`);
          } catch (e) {
            console.error('[session-bridge] Error writing message SSE:', e.message);
          }
        }
      }
    } catch (e) {
      console.error('[session-bridge] Message stream error:', e.message);
    }
  })();
}

// ---------------------------------------------------------------------------
// Conversation subscription
// ---------------------------------------------------------------------------
function subscribeToConversations() {
  (async () => {
    try {
      for await (const conv of client.conversations()) {
        if (currentConvSseClient) {
          try {
            currentConvSseClient.write(`data: ${JSON.stringify({ type: 'conversation:updated', data: conv })}\n\n`);
          } catch (e) {
            console.error('[session-bridge] Error writing conversation SSE:', e.message);
          }
        }
      }
    } catch (e) {
      console.error('[session-bridge] Conversation stream error:', e.message);
    }
  })();
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------
function sendReadyEvent() {
  if (currentSseClient) {
    try {
      currentSseClient.write(
        `data: ${JSON.stringify({ type: 'ready', data: { sessionId: client.getSessionId() } })}\n\n`
      );
    } catch (e) {
      console.error('[session-bridge] Error sending ready SSE:', e.message);
    }
  }
}

function initSseResponse(res) {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });
}

// ---------------------------------------------------------------------------
// JSON response helpers
// ---------------------------------------------------------------------------
function jsonOk(res, data, status = 200) {
  const body = JSON.stringify(data);
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(body);
}

function jsonError(res, message, status = 500) {
  const body = JSON.stringify({ error: message });
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(body);
}

// Parse POST body as JSON
function parseBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        resolve(JSON.parse(body || '{}'));
      } catch (e) {
        reject(new Error('Invalid JSON body'));
      }
    });
    req.on('error', reject);
  });
}

// ---------------------------------------------------------------------------
// HTTP request router
// ---------------------------------------------------------------------------
async function requestHandler(req, res) {
  const parsed = new URL(req.url, 'http://x');
  const pathname = parsed.pathname;
  const method = req.method;

  try {
    // -----------------------------------------------------------------------
    // GET /health
    // -----------------------------------------------------------------------
    if (method === 'GET' && pathname === '/health') {
      if (isReady) {
        return jsonOk(res, { status: 'ready', sessionId: client.getSessionId() });
      } else {
        return jsonOk(res, { status: 'starting' });
      }
    }

    // -----------------------------------------------------------------------
    // GET /session-id
    // -----------------------------------------------------------------------
    if (method === 'GET' && pathname === '/session-id') {
      const sessionId = client.getSessionId();
      if (!sessionId) return jsonError(res, 'Session ID not available yet', 503);
      return jsonOk(res, { sessionId });
    }

    // -----------------------------------------------------------------------
    // GET /events  — primary SSE stream (messages + ready)
    // -----------------------------------------------------------------------
    if (method === 'GET' && pathname === '/events') {
      // Disconnect any existing SSE client
      if (currentSseClient) {
        try { currentSseClient.end(); } catch (_) {}
      }
      currentSseClient = res;

      initSseResponse(res);

      // Send ready immediately if already bootstrapped
      if (isReady) {
        res.write(
          `data: ${JSON.stringify({ type: 'ready', data: { sessionId: client.getSessionId() } })}\n\n`
        );
      }

      // Keepalive ping every 30 s
      const pingInterval = setInterval(() => {
        try { res.write(': ping\n\n'); } catch (_) {}
      }, 30000);

      req.on('close', () => {
        clearInterval(pingInterval);
        if (currentSseClient === res) currentSseClient = null;
      });

      // Do NOT call res.end() — keep the stream open
      return;
    }

    // -----------------------------------------------------------------------
    // GET /conversations/stream  — SSE stream of conversation:updated events
    // -----------------------------------------------------------------------
    if (method === 'GET' && pathname === '/conversations/stream') {
      if (currentConvSseClient) {
        try { currentConvSseClient.end(); } catch (_) {}
      }
      currentConvSseClient = res;

      initSseResponse(res);

      const pingInterval = setInterval(() => {
        try { res.write(': ping\n\n'); } catch (_) {}
      }, 30000);

      req.on('close', () => {
        clearInterval(pingInterval);
        if (currentConvSseClient === res) currentConvSseClient = null;
      });

      return;
    }

    // -----------------------------------------------------------------------
    // GET /conversations
    // -----------------------------------------------------------------------
    if (method === 'GET' && pathname === '/conversations') {
      const convs = await client.getConversations();
      return jsonOk(res, convs);
    }

    // -----------------------------------------------------------------------
    // GET /messages/:conversationId
    // -----------------------------------------------------------------------
    const messagesMatch = pathname.match(/^\/messages\/(.+)$/);
    if (method === 'GET' && messagesMatch) {
      const conversationId = decodeURIComponent(messagesMatch[1]);
      const limit = parsed.searchParams.has('limit')
        ? parseInt(parsed.searchParams.get('limit'), 10)
        : undefined;
      const opts = limit !== undefined ? { limit } : {};
      const messages = await client.getMessages(conversationId, opts);
      return jsonOk(res, messages);
    }

    // -----------------------------------------------------------------------
    // POST /send
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/send') {
      const data = await parseBody(req);
      const { to, body: msgBody, attachments, quote, expireTimer } = data;
      if (!to || msgBody === undefined) {
        return jsonError(res, 'Missing required fields: to, body', 400);
      }
      const options = {};
      if (attachments)  options.attachments  = attachments;
      if (quote)        options.quote        = quote;
      if (expireTimer !== undefined) options.expireTimer = expireTimer;
      const msgId = await client.sendMessage(to, msgBody, options);
      return jsonOk(res, { success: true, id: msgId });
    }

    // -----------------------------------------------------------------------
    // POST /send-typing
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/send-typing') {
      const data = await parseBody(req);
      const { to, isTyping } = data;
      if (!to || isTyping === undefined) {
        return jsonError(res, 'Missing required fields: to, isTyping', 400);
      }
      await client.setTyping(to, isTyping);
      return jsonOk(res, { success: true });
    }

    // -----------------------------------------------------------------------
    // POST /accept-contact
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/accept-contact') {
      const data = await parseBody(req);
      const { sessionId } = data;
      if (!sessionId) return jsonError(res, 'Missing required field: sessionId', 400);
      await client.acceptContactRequest(sessionId);
      return jsonOk(res, { success: true });
    }

    // -----------------------------------------------------------------------
    // POST /download-attachment
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/download-attachment') {
      const data = await parseBody(req);
      const { attachment, destDir } = data;
      if (!attachment) return jsonError(res, 'Missing required field: attachment', 400);
      const filePath = await client.downloadAttachment(attachment, destDir);
      return jsonOk(res, { path: filePath });
    }

    // -----------------------------------------------------------------------
    // POST /react
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/react') {
      const data = await parseBody(req);
      const { conversationId, messageDbId, emoji } = data;
      if (!conversationId || messageDbId === undefined || !emoji) {
        return jsonError(res, 'Missing required fields: conversationId, messageDbId, emoji', 400);
      }
      await client.sendReaction(conversationId, messageDbId, emoji);
      return jsonOk(res, { success: true });
    }

    // -----------------------------------------------------------------------
    // POST /create-group
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/create-group') {
      const data = await parseBody(req);
      const { name, members } = data;
      if (!name || !Array.isArray(members)) {
        return jsonError(res, 'Missing required fields: name, members (array)', 400);
      }
      const groupId = await client.createGroup(name, members);
      return jsonOk(res, { groupId });
    }

    // -----------------------------------------------------------------------
    // POST /add-group-members
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/add-group-members') {
      const data = await parseBody(req);
      const { groupId, sessionIds, withHistory } = data;
      if (!groupId || !Array.isArray(sessionIds)) {
        return jsonError(res, 'Missing required fields: groupId, sessionIds (array)', 400);
      }
      const opts = withHistory !== undefined ? { withHistory } : {};
      await client.addGroupMembers(groupId, sessionIds, opts);
      return jsonOk(res, { success: true });
    }

    // -----------------------------------------------------------------------
    // POST /remove-group-members
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/remove-group-members') {
      const data = await parseBody(req);
      const { groupId, sessionIds, alsoRemoveMessages } = data;
      if (!groupId || !Array.isArray(sessionIds)) {
        return jsonError(res, 'Missing required fields: groupId, sessionIds (array)', 400);
      }
      const opts = alsoRemoveMessages !== undefined ? { alsoRemoveMessages } : {};
      await client.removeGroupMembers(groupId, sessionIds, opts);
      return jsonOk(res, { success: true });
    }

    // -----------------------------------------------------------------------
    // POST /promote-group-members
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/promote-group-members') {
      const data = await parseBody(req);
      const { groupId, memberIds } = data;
      if (!groupId || !Array.isArray(memberIds)) {
        return jsonError(res, 'Missing required fields: groupId, memberIds (array)', 400);
      }
      await client.promoteGroupMembers(groupId, memberIds);
      return jsonOk(res, { success: true });
    }

    // -----------------------------------------------------------------------
    // POST /leave-group
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/leave-group') {
      const data = await parseBody(req);
      const { groupId } = data;
      if (!groupId) return jsonError(res, 'Missing required field: groupId', 400);
      await client.leaveGroup(groupId);
      return jsonOk(res, { success: true });
    }

    // -----------------------------------------------------------------------
    // POST /block-contact
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/block-contact') {
      const data = await parseBody(req);
      const { sessionId } = data;
      if (!sessionId) return jsonError(res, 'Missing required field: sessionId', 400);
      await client.blockContact(sessionId);
      return jsonOk(res, { success: true });
    }

    // -----------------------------------------------------------------------
    // POST /unblock-contact
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/unblock-contact') {
      const data = await parseBody(req);
      const { sessionId } = data;
      if (!sessionId) return jsonError(res, 'Missing required field: sessionId', 400);
      await client.unblockContact(sessionId);
      return jsonOk(res, { success: true });
    }

    // -----------------------------------------------------------------------
    // POST /set-display-name
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/set-display-name') {
      const data = await parseBody(req);
      const { name } = data;
      if (!name) return jsonError(res, 'Missing required field: name', 400);
      await client.setDisplayName(name);
      return jsonOk(res, { success: true });
    }

    // -----------------------------------------------------------------------
    // POST /set-display-image
    // -----------------------------------------------------------------------
    if (method === 'POST' && pathname === '/set-display-image') {
      const data = await parseBody(req);
      const { imagePath } = data;
      if (!imagePath) return jsonError(res, 'Missing required field: imagePath', 400);
      const imageBuffer = fs.readFileSync(imagePath);
      await client.setDisplayImage(imageBuffer);
      return jsonOk(res, { success: true });
    }

    // -----------------------------------------------------------------------
    // 404 fallthrough
    // -----------------------------------------------------------------------
    return jsonError(res, `Not found: ${method} ${pathname}`, 404);

  } catch (err) {
    console.error('[session-bridge] Request error:', err.message);
    try {
      jsonError(res, err.message || 'Internal server error', 500);
    } catch (_) {
      // Response may already be partially written (e.g. SSE stream), ignore
    }
  }
}

// ---------------------------------------------------------------------------
// Graceful shutdown
// ---------------------------------------------------------------------------
async function shutdown() {
  console.error('[session-bridge] Shutting down...');
  try {
    await client.shutdown();
  } catch (e) {
    console.error('[session-bridge] Error during shutdown:', e.message);
  }
  process.exit(0);
}

process.on('SIGTERM', shutdown);
process.on('SIGINT',  shutdown);
