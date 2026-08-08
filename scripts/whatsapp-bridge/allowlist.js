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
  // Ordinary inbound group messages reach this classifier after the bridge's
  // existing fromMe-group rejection. Keep their authorization ahead of media
  // extraction so blocked-group attachments cannot touch disk.
  if (isGroup) {
    if (!matchesInboundWhatsAppGroup({
      chatId,
      groupPolicy,
      groupAllowedUsers,
      sessionDir,
    })) {
      return { allowed: false, reason: 'group_policy_rejected_before_media' };
    }
    return { allowed: true };
  }

  if (!fromMe && dmPolicy !== 'pairing' && !matchesAllowedUser(senderId, allowedUsers, sessionDir)) {
    return { allowed: false, reason: 'allowlist_mismatch' };
  }

  return { allowed: true };
}

export function prepareInboundMediaDispatch({ downloadMedia, ...accessInput }) {
  const access = classifyInboundAccessBeforeMedia(accessInput);
  const guardedDownloadMedia = async (...args) => {
    if (!access?.allowed) {
      throw new Error(access?.reason || 'media_download_rejected_before_access');
    }
    return downloadMedia(...args);
  };

  return { access, downloadMedia: guardedDownloadMedia };
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
