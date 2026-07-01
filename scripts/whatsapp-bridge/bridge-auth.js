import { timingSafeEqual } from 'crypto';

function tokensEqual(provided, expected) {
  const providedBuffer = Buffer.from(String(provided || ''), 'utf8');
  const expectedBuffer = Buffer.from(String(expected || ''), 'utf8');
  if (providedBuffer.length !== expectedBuffer.length) {
    timingSafeEqual(expectedBuffer, expectedBuffer);
    return false;
  }
  return timingSafeEqual(providedBuffer, expectedBuffer);
}

export function createBridgeAuthMiddleware(expectedToken) {
  const configuredToken = String(expectedToken || '');

  return function bridgeAuth(req, res, next) {
    if (req.path === '/health') {
      return next();
    }
    if (!configuredToken) {
      return res.status(503).json({ error: 'Bridge auth token is not configured' });
    }

    const authorization = String(req.headers.authorization || '').trim();
    const prefix = 'Bearer ';
    if (!authorization.startsWith(prefix)) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const providedToken = authorization.slice(prefix.length);
    if (!tokensEqual(providedToken, configuredToken)) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    return next();
  };
}
