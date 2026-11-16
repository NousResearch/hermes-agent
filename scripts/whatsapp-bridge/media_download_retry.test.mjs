import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { createCipheriv, createHmac, randomBytes } from 'node:crypto';
import { downloadMediaMessage, getMediaKeys } from '@whiskeysockets/baileys';
import { extractBridgeEvent } from './bridge_helpers.js';
import { downloadWithRetry } from './media_download_retry.js';

const quiet = { delays: [0, 0], log: () => {} };

test('retries are bounded, transient-only, and do not log private error details', async () => {
  const transient = [
    new TypeError('terminated'),
    Object.assign(new Error('rate limited'), { output: { statusCode: 429 } }),
    Object.assign(new Error('unavailable'), { output: { statusCode: 503 } }),
    new Error('stream', { cause: Object.assign(new Error('socket'), { code: 'UND_ERR_SOCKET' }) }),
  ];
  for (const error of transient) {
    let attempts = 0;
    const waits = [], logs = [];
    const result = await downloadWithRetry(async () => {
      if (++attempts < 3) throw error;
      return 'complete';
    }, { wait: async ms => waits.push(ms), log: line => logs.push(line) });
    assert.equal(result, 'complete');
    assert.deepEqual(waits, [1000, 3000]);
    assert.equal(logs.length, 2);
    assert.ok(logs.every(line => !line.includes(error.message)));
    attempts = 0;
    await assert.rejects(downloadWithRetry(async () => { attempts++; throw error; }, quiet), e => e === error);
    assert.equal(attempts, 3);
  }
  for (const error of [
    Object.assign(new Error('forbidden'), { output: { statusCode: 403 } }),
    Object.assign(new Error('missing'), { output: { statusCode: 404 } }),
    Object.assign(new Error('expired'), { output: { statusCode: 410 } }),
    new Error('bad decrypt'),
  ]) {
    let attempts = 0;
    await assert.rejects(downloadWithRetry(async () => { attempts++; throw error; }, quiet), e => e === error);
    assert.equal(attempts, 1);
  }
  let attempts = 0;
  assert.equal(await downloadWithRetry(async () => { attempts++; return 'complete'; }, quiet), 'complete');
  assert.equal(attempts, 1);
});

test('real Baileys audio download recovers from a dropped connection and publishes once', { timeout: 10000 }, async () => {
  const mediaKey = randomBytes(32);
  const { cipherKey, iv, macKey } = await getMediaKeys(mediaKey, 'audio');
  const audio = randomBytes(755628);
  const cipher = createCipheriv('aes-256-cbc', cipherKey, iv);
  const ciphertext = Buffer.concat([cipher.update(audio), cipher.final()]);
  const mac = createHmac('sha256', macKey).update(iv).update(ciphertext).digest().subarray(0, 10);
  const encrypted = Buffer.concat([ciphertext, mac]);
  let requests = 0;
  let interruptNext = true;
  const server = createServer((_request, response) => {
    requests++;
    if (interruptNext) {
      interruptNext = false;
      response.destroy();
    } else {
      response.writeHead(200, { 'Content-Length': encrypted.length });
      response.end(encrypted);
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const msg = { key: { id: 'synthetic-test' }, message: { audioMessage: {
      url: `http://127.0.0.1:${server.address().port}/synthetic.enc`,
      mediaKey, mimetype: 'audio/ogg; codecs=opus', ptt: true,
    } } };
    const download = () => downloadMediaMessage(msg, 'buffer', {});
    await assert.rejects(download(), error => error.message === 'fetch failed');
    interruptNext = true;
    let writes = 0;
    const event = await extractBridgeEvent({
      msg, chatId: 'test@s.whatsapp.net', senderId: 'test@s.whatsapp.net', senderNumber: 'test',
      downloadMedia: () => downloadWithRetry(download, quiet),
      writeMediaFile: async ({ buffer }) => {
        writes++;
        assert.deepEqual(buffer, audio);
        return '/synthetic/audio.ogg';
      },
      cacheDirs: { audio: '/synthetic' },
    });
    assert.equal(writes, 1);
    assert.deepEqual(event.mediaUrls, ['/synthetic/audio.ogg']);
    assert.equal(event.mediaType, 'ptt');
    assert.ok(!event.body.includes('could not be downloaded'));
    assert.equal(requests, 3); // baseline fails; retry test fails once then succeeds
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
});
