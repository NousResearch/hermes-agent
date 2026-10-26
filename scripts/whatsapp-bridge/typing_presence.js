const VALID_TYPING_PRESENCE = new Set(['composing', 'paused']);

function normalizeTypingPresence(value, field) {
  if (typeof value !== 'string') {
    throw new TypeError(`${field} must be composing or paused`);
  }
  const normalized = value.trim().toLowerCase();
  if (!VALID_TYPING_PRESENCE.has(normalized)) {
    throw new TypeError(`${field} must be composing or paused`);
  }
  return normalized;
}

export function resolveTypingPresence(body = {}) {
  if (body === null || typeof body !== 'object' || Array.isArray(body)) {
    throw new TypeError('typing request body must be an object');
  }

  const hasState = Object.prototype.hasOwnProperty.call(body, 'state');
  const hasAction = Object.prototype.hasOwnProperty.call(body, 'action');
  const state = hasState ? normalizeTypingPresence(body.state, 'state') : null;
  const action = hasAction ? normalizeTypingPresence(body.action, 'action') : null;

  if (state && action && state !== action) {
    throw new TypeError('typing state and action must match');
  }
  return state || action || 'composing';
}