import { strict as assert } from 'node:assert';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import express from 'express';

import { registerAlbumRoute, sendAlbumMessageWithWatchdog } from './album_route.js';

function createQueue() {
  let tail = Promise.resolve();
  return fn => {
    const task = tail.then(fn, fn);
    tail = task.catch(() => {});
    return task;
  };
}

async function withServer(socket, fn) {
  const app = express();
  app.use(express.json());
  registerAlbumRoute(app, {
    getSocket: () => socket,
    getConnectionState: () => 'connected',
    enqueueSend: createQueue(),
    trackSentMessageId: () => {},
    messageStore: { remember: () => {} },
  });
  const server = await new Promise(resolve => {
    const listening = app.listen(0, '127.0.0.1', () => resolve(listening));
  });
  try {
    const address = server.address();
    await fn(`http://127.0.0.1:${address.port}`);
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
}

{
  let fatalCalls = 0;
  let sendOptions;
  const socket = {
    sendMessage: async (_chatId, _payload, options) => {
      sendOptions = options;
      return new Promise(() => {});
    },
  };

  await assert.rejects(
    () => sendAlbumMessageWithWatchdog({
      socket,
      chatId: '15551234567@s.whatsapp.net',
      payload: { image: Buffer.from('stuck') },
      mediaUploadTimeoutMs: 10,
      watchdogTimeoutMs: 20,
      onFatalTimeout: async () => { fatalCalls += 1; },
    }),
    /timed out after 20ms/,
  );
  assert.equal(fatalCalls, 1);
  assert.deepEqual(sendOptions, { mediaUploadTimeoutMs: 10 });
  console.log('  ✓ hung album sends trigger the fatal watchdog within a bounded time');
}

{
  const dir = mkdtempSync(path.join(tmpdir(), 'hermes-wa-album-route-'));
  const first = path.join(dir, 'first.jpg');
  const second = path.join(dir, 'second.jpg');
  writeFileSync(first, Buffer.from('first'));
  writeFileSync(second, Buffer.from('second'));
  const calls = [];
  const parentKey = {
    id: 'parent',
    remoteJid: '15551234567@s.whatsapp.net',
  };
  const socket = {
    sendMessage: async (chatId, payload) => {
      calls.push({ chatId, payload });
      return { key: calls.length === 1 ? parentKey : { id: `child-${calls.length - 1}`, remoteJid: chatId } };
    },
  };

  await withServer(socket, async baseUrl => {
    const response = await fetch(`${baseUrl}/send-album`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        chatId: '15551234567@s.whatsapp.net',
        items: [
          { filePath: first, mediaType: 'image' },
          { filePath: second, mediaType: 'image' },
        ],
      }),
    });
    const body = await response.json();
    assert.equal(response.status, 200);
    assert.equal(body.status, 'success');
    assert.equal(body.parentMessageId, 'parent');
    assert.deepEqual(body.childMessageIds, ['child-1', 'child-2']);
    assert.equal(calls.length, 3);
    assert.deepEqual(calls[0].payload.album, {
      expectedImageCount: 2,
      expectedVideoCount: 0,
    });
    assert.deepEqual(calls[1].payload.albumParentKey, parentKey);
  });
  console.log('  ✓ /send-album sends a native parent and associated children over HTTP');
}

{
  let sends = 0;
  const socket = {
    sendMessage: async () => {
      sends += 1;
      return { key: { id: 'unexpected' } };
    },
  };
  await withServer(socket, async baseUrl => {
    const response = await fetch(`${baseUrl}/send-album`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        chatId: '15551234567@s.whatsapp.net',
        items: [
          { filePath: '/definitely/missing/one.jpg', mediaType: 'image' },
          { filePath: '/definitely/missing/two.jpg', mediaType: 'image' },
        ],
      }),
    });
    const body = await response.json();
    assert.equal(response.status, 400);
    assert.equal(body.attempted, false);
    assert.equal(body.status, 'validation_error');
    assert.equal(sends, 0);
  });
  console.log('  ✓ missing files fail preflight before the album parent is attempted');
}

console.log('\n✅ All WhatsApp album route integration tests passed.');
