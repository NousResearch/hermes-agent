import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildSentryOptions,
  captureBridgeException,
  sentryDsnFromEnv,
  sanitizeBridgeContext,
} from './sentry.js';

test('sentryDsnFromEnv accepts SENTRY_DSN and HERMES_SENTRY_DSN', () => {
  assert.equal(sentryDsnFromEnv({ SENTRY_DSN: 'https://public@sentry.example/1' }), 'https://public@sentry.example/1');
  assert.equal(sentryDsnFromEnv({ HERMES_SENTRY_DSN: 'https://public@sentry.example/2' }), 'https://public@sentry.example/2');
  assert.equal(sentryDsnFromEnv({}), '');
});

test('buildSentryOptions tags events as WhatsApp bridge without leaking defaults', () => {
  const options = buildSentryOptions({
    SENTRY_DSN: 'https://public@sentry.example/1',
    SENTRY_ENVIRONMENT: 'staging',
    SENTRY_RELEASE: 'hermes@1.2.3',
  });

  assert.equal(options.dsn, 'https://public@sentry.example/1');
  assert.equal(options.environment, 'staging');
  assert.equal(options.release, 'hermes@1.2.3');
  assert.deepEqual(options.initialScope.tags, {
    platform: 'whatsapp',
    component: 'whatsapp-bridge',
    runtime: 'node',
  });
});

test('sanitizeBridgeContext keeps operational metadata and redacts message identifiers', () => {
  assert.deepEqual(
    sanitizeBridgeContext({
      operation: 'send',
      endpoint: '/send',
      chatId: '5511999999999@s.whatsapp.net',
      senderId: '5511888888888@s.whatsapp.net',
      message: 'secret body',
      filePath: '/home/user/private.pdf',
      mediaType: 'document',
      status: 500,
    }),
    {
      operation: 'send',
      endpoint: '/send',
      chatId: '[redacted]',
      senderId: '[redacted]',
      message: '[redacted]',
      filePath: '[redacted]',
      mediaType: 'document',
      status: 500,
    },
  );
});

test('captureBridgeException reports sanitized context to the supplied Sentry client', () => {
  const calls = [];
  const fakeSentry = {
    captureException(error, configureScope) {
      const scope = {
        tags: {},
        contexts: {},
        setTag(key, value) {
          this.tags[key] = value;
        },
        setContext(key, value) {
          this.contexts[key] = value;
        },
      };
      configureScope(scope);
      calls.push({ error, scope });
      return 'event-id';
    },
  };

  const error = new Error('boom');
  const eventId = captureBridgeException(error, {
    operation: 'download_media',
    chatId: '5511999999999@s.whatsapp.net',
    mediaType: 'image',
  }, fakeSentry);

  assert.equal(eventId, 'event-id');
  assert.equal(calls.length, 1);
  assert.equal(calls[0].error, error);
  assert.deepEqual(calls[0].scope.tags, {
    platform: 'whatsapp',
    component: 'whatsapp-bridge',
  });
  assert.deepEqual(calls[0].scope.contexts.whatsapp_bridge, {
    operation: 'download_media',
    chatId: '[redacted]',
    mediaType: 'image',
  });
});

test('captureBridgeException is a no-op when Sentry is not configured', () => {
  assert.equal(captureBridgeException(new Error('ignored'), { operation: 'test' }, null), null);
});
