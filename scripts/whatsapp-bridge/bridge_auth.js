import { existsSync, readFileSync } from 'fs';
import path from 'path';
import { timingSafeEqual } from 'crypto';

export function defaultBridgeTokenPath(env = process.env) {
  const hermesHome = env.HERMES_HOME
    || path.join(env.LOCALAPPDATA || path.join(env.USERPROFILE || '.', 'AppData', 'Local'), 'hermes');
  return path.join(hermesHome, 'secrets', 'whatsapp_bridge_token');
}

export function loadBridgeToken({
  tokenPath = defaultBridgeTokenPath(),
  envToken = process.env.HERMES_WA_BRIDGE_TOKEN || '',
} = {}) {
  const fromEnv = String(envToken || '').trim();
  if (fromEnv) return fromEnv;
  if (!existsSync(tokenPath)) {
    throw new Error(`WhatsApp bridge token is not configured at ${tokenPath}`);
  }
  const fromFile = readFileSync(tokenPath, 'utf8').trim();
  if (!fromFile) {
    throw new Error(`WhatsApp bridge token is empty at ${tokenPath}`);
  }
  return fromFile;
}

function constantTimeEqual(left, right) {
  const a = Buffer.from(String(left || ''), 'utf8');
  const b = Buffer.from(String(right || ''), 'utf8');
  if (a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}

export function createBridgeAuthMiddleware(expectedToken) {
  const token = String(expectedToken || '').trim();
  if (!token) throw new Error('WhatsApp bridge authentication token is required');
  return (req, res, next) => {
    const header = String(req.get('authorization') || '');
    const supplied = header.startsWith('Bearer ') ? header.slice(7) : '';
    if (!constantTimeEqual(supplied, token)) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
    return next();
  };
}
