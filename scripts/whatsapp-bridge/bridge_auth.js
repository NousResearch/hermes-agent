import { createHmac, timingSafeEqual } from 'node:crypto';

function constantTimeEqual(left, right) {
  const leftBuffer = Buffer.from(String(left || ''), 'utf8');
  const rightBuffer = Buffer.from(String(right || ''), 'utf8');
  return leftBuffer.length === rightBuffer.length && timingSafeEqual(leftBuffer, rightBuffer);
}

export function bridgeAuthProof(token, challenge) {
  return createHmac('sha256', String(token)).update(String(challenge)).digest('hex');
}

export function createBridgeAuthMiddleware(token) {
  const expected = `Bearer ${token}`;
  return (req, res, next) => {
    if (!token || !constantTimeEqual(req.headers.authorization, expected)) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
    next();
  };
}
