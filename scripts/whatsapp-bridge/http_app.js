import express from 'express';

const ACCEPTED_HOST_VALUES = new Set([
  'localhost',
  '127.0.0.1',
  '[::1]',
  '::1',
]);

export function createBridgeApp() {
  const app = express();
  app.disable('x-powered-by');
  app.use(express.json());

  // The bridge is loopback-only, but hostile DNS can still resolve to
  // 127.0.0.1. Reject non-loopback Host values before endpoint dispatch.
  app.use((req, res, next) => {
    const raw = (req.headers.host || '').trim();
    if (!raw) {
      return res.status(400).json({ error: 'Missing Host header' });
    }

    const hostOnly = (raw.includes(':')
      ? raw.substring(0, raw.lastIndexOf(':'))
      : raw
    ).replace(/^\[|\]$/g, '').toLowerCase();
    if (!ACCEPTED_HOST_VALUES.has(hostOnly)) {
      return res.status(400).json({
        error: 'Invalid Host header. Bridge accepts loopback hosts only.',
      });
    }
    next();
  });

  return app;
}
