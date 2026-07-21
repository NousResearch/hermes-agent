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

export function isGroupChatAllowed(chatId, groupPolicy, allowedGroups, sessionDir) {
  const policy = String(groupPolicy || '').trim().toLowerCase();
  if (policy === 'open') {
    return true;
  }
  if (policy === 'allowlist') {
    return matchesAllowedUser(chatId, allowedGroups, sessionDir);
  }
  return false;
}

export function classifyInboundAccessBeforeMedia({
  isGroup,
  fromMe,
  chatId,
  senderId,
  dmPolicy,
  allowedUsers,
  groupPolicy,
  groupAllowedUsers,
  sessionDir,
}) {
  // Group authorization applies to every message that can reach extraction,
  // including owner-authored messages forwarded from bot mode. Restricting
  // this check to !fromMe would let blocked-group media touch disk.
  if (isGroup) {
    if (!isGroupChatAllowed(chatId, groupPolicy, groupAllowedUsers, sessionDir)) {
      return { allowed: false, reason: 'group_policy_rejected_before_media' };
    }
    return { allowed: true };
  }

  if (!fromMe && dmPolicy !== 'pairing' && !matchesAllowedUser(senderId, allowedUsers, sessionDir)) {
    return { allowed: false, reason: 'allowlist_mismatch' };
  }

  return { allowed: true };
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
