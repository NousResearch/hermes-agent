const REDACTED = '[redacted]';
const REDACT_KEYS = new Set([
  'body',
  'caption',
  'chatId',
  'fileName',
  'filePath',
  'message',
  'messageId',
  'quotedMessageId',
  'quotedParticipant',
  'quotedRemoteJid',
  'senderId',
  'senderName',
]);

let sentryClient = null;

export function sentryDsnFromEnv(env = process.env) {
  return String(env.SENTRY_DSN || env.HERMES_SENTRY_DSN || '').trim();
}

export function buildSentryOptions(env = process.env) {
  const dsn = sentryDsnFromEnv(env);
  const options = {
    dsn,
    environment: env.SENTRY_ENVIRONMENT || env.HERMES_ENV || env.NODE_ENV || 'production',
    initialScope: {
      tags: {
        platform: 'whatsapp',
        component: 'whatsapp-bridge',
        runtime: 'node',
      },
    },
  };

  if (env.SENTRY_RELEASE) {
    options.release = env.SENTRY_RELEASE;
  }

  const tracesSampleRate = env.SENTRY_TRACES_SAMPLE_RATE;
  if (tracesSampleRate !== undefined && tracesSampleRate !== '') {
    const parsed = Number.parseFloat(tracesSampleRate);
    if (Number.isFinite(parsed)) {
      options.tracesSampleRate = parsed;
    }
  }

  return options;
}

export async function initSentryForBridge({ env = process.env, clientLoader, logger = console } = {}) {
  if (!sentryDsnFromEnv(env)) {
    return null;
  }

  try {
    const Sentry = clientLoader ? await clientLoader() : await import('@sentry/node');
    const client = Sentry.default || Sentry;
    client.init(buildSentryOptions(env));
    sentryClient = client;
    logger.log?.('[bridge] Sentry error tracking enabled');
    return client;
  } catch (err) {
    logger.warn?.('[bridge] Failed to initialize Sentry error tracking:', err?.message || err);
    return null;
  }
}

export function sanitizeBridgeContext(context = {}) {
  const sanitized = {};
  for (const [key, value] of Object.entries(context || {})) {
    if (REDACT_KEYS.has(key)) {
      sanitized[key] = REDACTED;
    } else if (Array.isArray(value)) {
      sanitized[key] = value.map((item) => {
        if (typeof item === 'string') return item.length > 120 ? `${item.slice(0, 117)}...` : item;
        if (item && typeof item === 'object') return sanitizeBridgeContext(item);
        return item;
      });
    } else if (value && typeof value === 'object') {
      sanitized[key] = sanitizeBridgeContext(value);
    } else {
      sanitized[key] = value;
    }
  }
  return sanitized;
}

export function captureBridgeException(error, context = {}, client = sentryClient) {
  if (!client || typeof client.captureException !== 'function') {
    return null;
  }

  return client.captureException(error, (scope) => {
    scope.setTag?.('platform', 'whatsapp');
    scope.setTag?.('component', 'whatsapp-bridge');
    scope.setContext?.('whatsapp_bridge', sanitizeBridgeContext(context));
    return scope;
  });
}
