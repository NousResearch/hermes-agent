import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';

import { buildMediaPayload, inferMediaType } from './media-payload.js';

function withTempDir(fn) {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-media-'));
  try {
    return fn(dir);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

for (const ext of ['ogg', 'opus']) {
  test(`buildMediaPayload sends .${ext} audio as a native voice note without ffmpeg`, () => {
    withTempDir((dir) => {
      const filePath = path.join(dir, `reply.${ext}`);
      const original = Buffer.from(`voice-${ext}`);
      writeFileSync(filePath, original);

      const { payload, warning } = buildMediaPayload({
        filePath,
        exec: () => {
          throw new Error('ffmpeg should not run');
        },
      });

      assert.equal(warning, null);
      assert.deepEqual(payload.audio, original);
      assert.equal(payload.mimetype, 'audio/ogg; codecs=opus');
      assert.equal(payload.ptt, true);
    });
  });
}

test('buildMediaPayload converts non-Opus audio to a native voice note when ffmpeg succeeds', () => {
  withTempDir((dir) => {
    const filePath = path.join(dir, 'reply.mp3');
    const tmpPath = path.join(dir, 'reply-converted.ogg');
    const converted = Buffer.from('converted-opus');
    writeFileSync(filePath, Buffer.from('original-mp3'));
    let execCommand = '';
    let cleaned = false;

    const { payload, warning } = buildMediaPayload({
      filePath,
      makeTempPath: () => tmpPath,
      exec: (command, options) => {
        execCommand = command;
        assert.deepEqual(options, { timeout: 30000, stdio: 'pipe' });
        writeFileSync(tmpPath, converted);
      },
      remove: (target) => {
        assert.equal(target, tmpPath);
        cleaned = true;
      },
    });

    assert.equal(warning, null);
    assert.match(execCommand, /^ffmpeg -y -i /);
    assert.match(execCommand, / -ar 48000 /);
    assert.match(execCommand, / -ac 1 /);
    assert.match(execCommand, / -c:a libopus /);
    assert.match(execCommand, / -b:a 32k /);
    assert.match(execCommand, / -vbr on /);
    assert.match(execCommand, / -application voip /);
    assert.deepEqual(payload.audio, converted);
    assert.equal(payload.mimetype, 'audio/ogg; codecs=opus');
    assert.equal(payload.ptt, true);
    assert.equal(cleaned, true);
  });
});

test('buildMediaPayload falls back to original audio when ffmpeg conversion fails', () => {
  withTempDir((dir) => {
    const filePath = path.join(dir, 'reply.mp3');
    const original = Buffer.from('original-mp3');
    writeFileSync(filePath, original);

    const { payload, warning } = buildMediaPayload({
      filePath,
      makeTempPath: () => path.join(dir, 'reply-converted.ogg'),
      exec: () => {
        throw new Error('missing ffmpeg');
      },
    });

    assert.equal(warning, 'missing ffmpeg');
    assert.deepEqual(payload.audio, original);
    assert.equal(payload.mimetype, 'audio/mpeg');
    assert.equal(payload.ptt, false);
  });
});

test('inferMediaType classifies WhatsApp voice-note formats as audio', () => {
  assert.equal(inferMediaType('ogg'), 'audio');
  assert.equal(inferMediaType('opus'), 'audio');
});
