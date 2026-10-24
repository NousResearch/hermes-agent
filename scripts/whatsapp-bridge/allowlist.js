import path from 'path';
import { existsSync, readFileSync } from 'fs';

export function normalizeWhatsAppIdentifier(value) {
  return String(value || '')
    .trim()
    .replace(/:.*@/, '@')
    .replace(/@.*/, '')
    .replace(/^\+/, '');
}

export function parseAllowedUsers(rawValue) {
  return new Set(
    String(rawValue || '')
      .split(',')
      .map((value) => normalizeWhatsAppIdentifier(value))
      .filter(Boolean)
  );
}

function readMappingFile(sessionDir, identifier, suffix = '') {
  const filePath = path.join(sessionDir, `lid-mapping-${identifier}${suffix}.json`);
  if (!existsSync(filePath)) {
    return null;
  }

  try {
    const parsed = JSON.parse(readFileSync(filePath, 'utf8'));
    const normalized = normalizeWhatsAppIdentifier(parsed);
    return normalized || null;
  } catch {
    return null;
  }
}

export function expandWhatsAppIdentifiers(identifier, sessionDir) {
  const normalized = normalizeWhatsAppIdentifier(identifier);
  if (!normalized) {
    return new Set();
  }

  // Walk both phone->LID and LID->phone mapping files so allowlists can use
  // either form transparently in bot mode.
  const resolved = new Set();
  const queue = [normalized];

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || resolved.has(current)) {
      continue;
    }

    resolved.add(current);

    for (const suffix of ['', '_reverse']) {
      const mapped = readMappingFile(sessionDir, current, suffix);
      if (mapped && !resolved.has(mapped)) {
        queue.push(mapped);
      }
    }
  }

  return resolved;
}

export function matchesAllowedSender(senderId, senderAltId, allowedUsers, sessionDir) {
  // WhatsApp Multi-Device can expose a first-contact sender as an opaque LID
  // before it persists LID mapping files. Baileys supplies the same sender's
  // phone JID separately (key.remoteJidAlt for DMs, key.participantAlt in
  // groups), so consult it as an additional alias without changing the
  // allowlist itself (#63415, #72529).
  return matchesAllowedUser(senderId, allowedUsers, sessionDir)
    || matchesAllowedUser(senderAltId, allowedUsers, sessionDir);
}

export function matchesInboundWhatsAppGroup({
  chatId,
  groupPolicy,
  groupAllowedUsers,
  sessionDir,
}) {
  if (groupPolicy === 'disabled' || groupPolicy === 'pairing') {
    return false;
  }
  if (groupPolicy === 'allowlist') {
    return matchesAllowedUser(chatId, groupAllowedUsers, sessionDir);
  }
  return groupPolicy === 'open';
}

/**
 * True when ``identifier`` names one of the account's OWN identities, so a
 * "Message yourself" self-chat bound to it must be accepted (self-chat mode).
 *
 *   * ``accountId`` / ``accountLid`` are the current ``sock.user`` identity.
 *   * ``selfIdentifiers`` are the identities the account's own operator
 *     declares: self-chat mode seeds ``WHATSAPP_ALLOWED_USERS`` with the paired
 *     number, and an operator who changed numbers can keep the previous one
 *     there. Each declared identity is expanded through its own
 *     ``lid-mapping-*.json`` pair, which only adds that identity's other form.
 *     ``*`` means "allow everyone" and never means "this is me".
 *
 * The session's mapping files are deliberately NOT searched for the incoming
 * identifier on their own: they map every known CONTACT, so treating their LIDs
 * as "self" would let a stranger's DM through the self-chat guard and reopen
 * the pairing-code reply (#8389).
 */
export function isSelfChatIdentity(identifier, {
  accountId,
  accountLid,
  selfIdentifiers,
  sessionDir,
} = {}) {
  const normalized = normalizeWhatsAppIdentifier(identifier);
  if (!normalized) {
    return false;
  }

  for (const own of [accountId, accountLid]) {
    const normalizedOwn = normalizeWhatsAppIdentifier(own);
    if (normalizedOwn && normalizedOwn === normalized) {
      return true;
    }
  }

  for (const declared of selfIdentifiers || []) {
    if (!declared || declared === '*') {
      continue;
    }
    if (expandWhatsAppIdentifiers(declared, sessionDir).has(normalized)) {
      return true;
    }
  }

  return false;
}

export function matchesAllowedUser(senderId, allowedUsers, sessionDir) {
  // Empty allowlist = NO ONE allowed (secure default, #8389).  Operators
  // who want an open bot must set ``WHATSAPP_ALLOWED_USERS=*`` explicitly.
  // Previous behaviour (empty → return true) let any stranger DM the
  // bridge and trigger a Python-side pairing-code reply.
  if (!allowedUsers || allowedUsers.size === 0) {
    return false;
  }

  // "*" means allow everyone (consistent with SIGNAL_GROUP_ALLOWED_USERS)
  if (allowedUsers.has('*')) {
    return true;
  }

  const aliases = expandWhatsAppIdentifiers(senderId, sessionDir);
  for (const alias of aliases) {
    if (allowedUsers.has(alias)) {
      return true;
    }
  }

  return false;
}
