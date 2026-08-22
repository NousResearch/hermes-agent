import path from 'path';
import { closeSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync, writeFileSync } from 'fs';
import { createHash, randomBytes, timingSafeEqual } from 'crypto';
import {
  expandWhatsAppIdentifiers,
  normalizeWhatsAppIdentifier,
} from './allowlist.js';

export const MIME_MAP = {
  jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png',
  webp: 'image/webp', gif: 'image/gif',
  mp4: 'video/mp4', mov: 'video/quicktime', avi: 'video/x-msvideo',
  mkv: 'video/x-matroska', '3gp': 'video/3gpp',
  pdf: 'application/pdf',
  doc: 'application/msword',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
};

export function normalizeWhatsAppId(value) {
  if (!value) return '';
  return String(value).replace(':', '@');
}

const DELIVERY_STATUS = new Map([
  [2, 'sent'],
  [3, 'delivered'],
  [4, 'read'],
  [5, 'read'],
  ['SERVER_ACK', 'sent'],
  ['DELIVERY_ACK', 'delivered'],
  ['READ', 'read'],
  ['PLAYED', 'read'],
]);

function validOutboundReceiptKey(key) {
  return key?.fromMe === true
    && typeof key.id === 'string'
    && /^[A-Za-z0-9_-]{1,191}$/.test(key.id)
    && !(typeof key.remoteJid === 'string'
      && (key.remoteJid.endsWith('@g.us') || key.remoteJid.endsWith('@broadcast')));
}

function timestampIso(value) {
  if (value === null || value === undefined) return null;
  const seconds = typeof value?.toNumber === 'function' ? value.toNumber() : Number(value);
  if (!Number.isFinite(seconds) || seconds <= 0) return null;
  try {
    return new Date(seconds * 1000).toISOString();
  } catch {
    return null;
  }
}

export function deliveryReceiptFromMessageUpdate({ key, update, now = () => new Date().toISOString() }) {
  if (!validOutboundReceiptKey(key)) return null;
  const rawStatus = update?.status;
  const numericStatus = typeof rawStatus === 'string' && /^\d+$/.test(rawStatus)
    ? Number(rawStatus)
    : rawStatus;
  const status = DELIVERY_STATUS.get(numericStatus);
  if (!status) return null;
  return { messageId: key.id, status, occurredAt: now() };
}

export function deliveryReceiptFromUserReceiptUpdate({
  key,
  receipt,
  now = () => new Date().toISOString(),
}) {
  if (!validOutboundReceiptKey(key) || !receipt || typeof receipt !== 'object') return null;
  const readAt = timestampIso(receipt.playedTimestamp) || timestampIso(receipt.readTimestamp);
  if (readAt) return { messageId: key.id, status: 'read', occurredAt: readAt };
  const deliveredAt = timestampIso(receipt.receiptTimestamp);
  const hasDeliveredDevice = Array.isArray(receipt.deliveredDeviceJid)
    && receipt.deliveredDeviceJid.length > 0;
  if (deliveredAt || hasDeliveredDevice) {
    return { messageId: key.id, status: 'delivered', occurredAt: deliveredAt || now() };
  }
  return null;
}

export function acknowledgeDeliveryReceipts(queue, acknowledgements) {
  const keys = new Set();
  for (const acknowledgement of acknowledgements || []) {
    const messageId = acknowledgement?.messageId;
    const status = acknowledgement?.status;
    if (typeof messageId !== 'string' || !/^[A-Za-z0-9_-]{1,191}$/.test(messageId)) continue;
    if (!['sent', 'delivered', 'read'].includes(status)) continue;
    keys.add(`${messageId}:${status}`);
  }
  let removed = 0;
  for (let index = queue.length - 1; index >= 0; index -= 1) {
    const receipt = queue[index];
    if (keys.has(`${receipt?.messageId}:${receipt?.status}`)) {
      queue.splice(index, 1);
      removed += 1;
    }
  }
  return removed;
}

const DELIVERY_STATUS_RANK = new Map([
  ['sent', 1],
  ['delivered', 2],
  ['read', 3],
]);

export function createDeliveryReceiptQueue({
  capacity = 1000,
  pageSize = 100,
  tombstoneRetentionMs = 30 * 24 * 60 * 60 * 1000,
  now = Date.now,
} = {}) {
  if (!Number.isInteger(capacity) || capacity < 1) throw new RangeError('capacity must be positive');
  if (!Number.isInteger(pageSize) || pageSize < 1 || pageSize > capacity) {
    throw new RangeError('pageSize must be within capacity');
  }
  if (!Number.isFinite(tombstoneRetentionMs) || tombstoneRetentionMs <= 0) {
    throw new RangeError('tombstoneRetentionMs must be positive');
  }
  if (typeof now !== 'function') throw new TypeError('now must be a function');
  const queue = [];
  const seen = new Set();
  const tombstones = new Map();

  function pruneExpiredTombstones(referenceTime = now()) {
    const cutoff = referenceTime - tombstoneRetentionMs;
    for (const [messageId, tombstone] of tombstones) {
      if (!Number.isFinite(tombstone?.acknowledgedAt) || tombstone.acknowledgedAt <= cutoff) {
        tombstones.delete(messageId);
      }
    }
  }

  function highestKnownRank(messageId) {
    let rank = 0;
    for (const receipt of queue) {
      if (receipt.messageId === messageId) {
        rank = Math.max(rank, DELIVERY_STATUS_RANK.get(receipt.status));
      }
    }
    rank = Math.max(rank, tombstones.get(messageId)?.rank || 0);
    return rank;
  }

  return {
    add(receipt) {
      const messageId = receipt?.messageId;
      const status = receipt?.status;
      if (typeof messageId !== 'string' || !/^[A-Za-z0-9_-]{1,191}$/.test(messageId)) return false;
      if (!DELIVERY_STATUS_RANK.has(status)) return false;
      if (typeof receipt.occurredAt !== 'string' || !receipt.occurredAt) return false;
      pruneExpiredTombstones();
      const key = `${messageId}:${status}`;
      if (seen.has(key)) return false;
      if (highestKnownRank(messageId) >= DELIVERY_STATUS_RANK.get(status)) return false;
      seen.add(key);
      queue.push(receipt);
      if (queue.length > capacity) {
        const evicted = queue.shift();
        seen.delete(`${evicted.messageId}:${evicted.status}`);
      }
      return true;
    },
    snapshot() {
      return queue.slice(0, pageSize);
    },
    acknowledge(acknowledgements) {
      const acknowledgedKeys = new Set();
      for (const acknowledgement of acknowledgements || []) {
        const messageId = acknowledgement?.messageId;
        const status = acknowledgement?.status;
        if (typeof messageId === 'string' && DELIVERY_STATUS_RANK.has(status)) {
          const key = `${messageId}:${status}`;
          if (seen.has(key)) acknowledgedKeys.add(key);
        }
      }
      const removedExact = acknowledgeDeliveryReceipts(queue, acknowledgements);
      const removedKeys = new Set(acknowledgedKeys);
      if (removedExact) {
        const acknowledgedRanks = new Map();
        for (const key of acknowledgedKeys) {
          const separator = key.lastIndexOf(':');
          const messageId = key.slice(0, separator);
          const rank = DELIVERY_STATUS_RANK.get(key.slice(separator + 1));
          acknowledgedRanks.set(messageId, Math.max(acknowledgedRanks.get(messageId) || 0, rank));
        }
        for (let index = queue.length - 1; index >= 0; index -= 1) {
          const receipt = queue[index];
          const acknowledgedRank = acknowledgedRanks.get(receipt.messageId) || 0;
          if (DELIVERY_STATUS_RANK.get(receipt.status) <= acknowledgedRank) {
            queue.splice(index, 1);
            removedKeys.add(`${receipt.messageId}:${receipt.status}`);
          }
        }
        const acknowledgedAt = now();
        for (const key of removedKeys) seen.delete(key);
        for (const [messageId, rank] of acknowledgedRanks) {
          const previousRank = tombstones.get(messageId)?.rank || 0;
          tombstones.set(messageId, { rank: Math.max(previousRank, rank), acknowledgedAt });
        }
      }
      return removedKeys.size;
    },
    size() {
      return queue.length;
    },
  };
}

export function ownerMessageTokenMatches(expectedSecret, authorizationHeader) {
  if (typeof expectedSecret !== 'string' || expectedSecret.length < 1) return false;
  if (typeof authorizationHeader !== 'string' || !authorizationHeader.startsWith('Bearer ')) {
    return false;
  }
  const expected = createHash('sha256').update(expectedSecret, 'utf8').digest();
  const supplied = createHash('sha256').update(authorizationHeader.slice(7), 'utf8').digest();
  return timingSafeEqual(expected, supplied);
}

export function providerSendIntentStatus({
  externalConsumer,
  sendIntent,
  expectedOwnerFenceSequence,
  expectedSecret,
  authorizationHeader,
} = {}) {
  if (!externalConsumer) return 'standard';
  if (sendIntent === 'human') {
    return ownerMessageTokenMatches(expectedSecret, authorizationHeader) ? 'human' : 'unauthorized';
  }
  if (sendIntent === 'automatic') {
    return Number.isSafeInteger(expectedOwnerFenceSequence) && expectedOwnerFenceSequence >= 0
      ? 'automatic' : 'invalid';
  }
  return 'invalid';
}

export function providerSendErrorResponse(error, messageIds = []) {
  const ids = Array.isArray(messageIds) ? messageIds.filter(id => typeof id === 'string' && id) : [];
  if (error?.code === 'OWNER_FENCED') {
    return {
      statusCode: 409,
      body: {
        error: error.message,
        code: 'OWNER_FENCED',
        retryable: false,
        partial: ids.length > 0,
        messageId: ids[ids.length - 1],
        messageIds: ids,
      },
    };
  }
  if (error?.code === 'SEND_QUEUE_EXPIRED' || error?.code === 'SEND_QUEUE_ABANDONED') {
    return {
      statusCode: 503,
      body: {
        error: error.message,
        code: error.code,
        retryable: true,
        partial: ids.length > 0,
        messageId: ids[ids.length - 1],
        messageIds: ids,
      },
    };
  }
  return {
    statusCode: 500,
    body: {
      error: error?.message || 'Provider send failed',
      partial: ids.length > 0,
      messageId: ids[ids.length - 1],
      messageIds: ids,
    },
  };
}

export function ownerMessageDeliveryMode(fromOwner, externalConsumer) {
  return fromOwner && externalConsumer ? 'external' : 'inbound';
}

export function ownerSendFenceStatus(queue, chatId, expectedSequence) {
  if (!queue || typeof queue.lastSequence !== 'function') return 'invalid';
  if (!Number.isSafeInteger(expectedSequence) || expectedSequence < 0) return 'invalid';
  if (typeof queue.hasUnresolvedFence === 'function' && queue.hasUnresolvedFence()) return 'fenced';
  return queue.lastSequence(chatId) === expectedSequence ? 'allowed' : 'fenced';
}

export function createSerializedSendQueue({ now = Date.now } = {}) {
  if (typeof now !== 'function') throw new TypeError('now must be a function');
  let queue = Promise.resolve();
  function enqueueSend(fn, {
    beforeRun,
    timeoutMs,
    timeoutError = () => new Error('send timed out'),
    queueDeadlineAt,
    queueError = () => Object.assign(
      new Error('send expired while waiting in the bridge queue'),
      { code: 'SEND_QUEUE_EXPIRED' },
    ),
    isAbandoned,
  } = {}) {
    let signalStart;
    const started = new Promise(resolve => { signalStart = resolve; });
    const run = async () => {
      try {
        if (typeof isAbandoned === 'function' && isAbandoned()) {
          throw Object.assign(
            new Error('request disconnected while waiting in the bridge queue'),
            { code: 'SEND_QUEUE_ABANDONED' },
          );
        }
        if (Number.isFinite(queueDeadlineAt) && now() >= queueDeadlineAt) throw queueError();
        if (beforeRun) await beforeRun();
        const providerPromise = Promise.resolve().then(fn);
        signalStart({ providerPromise });
        return await providerPromise;
      } catch (error) {
        signalStart({ error });
        throw error;
      }
    };
    const occupancy = queue.then(run, run);
    queue = occupancy.catch(() => {});
    if (!Number.isFinite(timeoutMs) || timeoutMs < 0) return occupancy;
    return started.then(({ providerPromise, error }) => {
      if (error) throw error;
      let timer;
      const timeoutPromise = new Promise((_, reject) => {
        timer = setTimeout(() => reject(timeoutError()), timeoutMs);
      });
      return Promise.race([providerPromise, timeoutPromise]).finally(() => clearTimeout(timer));
    });
  }
  return { enqueueSend };
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function validSequenceMap(value, keyPattern) {
  return isPlainObject(value) && Object.entries(value).every(([key, sequence]) => (
    keyPattern.test(key) && Number.isSafeInteger(sequence) && sequence > 0
  ));
}

function validateOwnerMessageState(parsed) {
  const invalid = () => { throw new Error('Invalid owner message queue state'); };
  if (!isPlainObject(parsed) || !Number.isSafeInteger(parsed.nextSequence)
    || parsed.nextSequence < 1 || !Array.isArray(parsed.entries)) invalid();

  const lastSequenceByChat = parsed.lastSequenceByChat ?? {};
  const unresolvedLidFences = parsed.unresolvedLidFences ?? {};
  const tombstones = parsed.tombstones ?? {};
  if (!validSequenceMap(lastSequenceByChat, /^\d{7,20}@(s\.whatsapp\.net|lid)$/)
    || !validSequenceMap(unresolvedLidFences, /^\d{7,20}@lid$/)
    || !isPlainObject(tombstones)) invalid();

  const messageIds = new Set();
  const sequences = new Set();
  let maximumSequence = 0;
  for (const entry of parsed.entries) {
    const event = entry?.event;
    const valid = isPlainObject(entry)
      && Number.isSafeInteger(entry.sequence) && entry.sequence > 0
      && isPlainObject(event)
      && typeof event.messageId === 'string' && /^[A-Za-z0-9_-]{1,191}$/.test(event.messageId)
      && typeof event.chatId === 'string' && /^\d{7,20}@(s\.whatsapp\.net|lid)$/.test(event.chatId)
      && typeof event.senderId === 'string' && event.senderId.length > 0
      && event.fromOwner === true
      && typeof event.type === 'string' && event.type.length > 0
      && typeof event.body === 'string'
      && Number.isFinite(event.timestamp) && event.timestamp > 0
      && event.ownerFenceSequence === entry.sequence
      && !messageIds.has(event.messageId)
      && !sequences.has(entry.sequence);
    if (!valid) invalid();
    messageIds.add(event.messageId);
    sequences.add(entry.sequence);
    maximumSequence = Math.max(maximumSequence, entry.sequence);
  }
  for (const [messageId, tombstone] of Object.entries(tombstones)) {
    const valid = /^[A-Za-z0-9_-]{1,191}$/.test(messageId)
      && !messageIds.has(messageId)
      && isPlainObject(tombstone)
      && Number.isSafeInteger(tombstone.sequence) && tombstone.sequence > 0
      && Number.isFinite(tombstone.acknowledgedAt) && tombstone.acknowledgedAt > 0;
    if (!valid) invalid();
    messageIds.add(messageId);
    maximumSequence = Math.max(maximumSequence, tombstone.sequence);
  }
  for (const sequence of Object.values(lastSequenceByChat)) {
    maximumSequence = Math.max(maximumSequence, sequence);
  }
  for (const sequence of Object.values(unresolvedLidFences)) {
    maximumSequence = Math.max(maximumSequence, sequence);
  }
  if (parsed.nextSequence <= maximumSequence) invalid();

  return { ...parsed, lastSequenceByChat, unresolvedLidFences, tombstones };
}

export function createOwnerMessageQueue({
  directory,
  pageSize = 100,
  tombstoneRetentionMs = 30 * 24 * 60 * 60 * 1000,
  now = Date.now,
} = {}) {
  if (typeof directory !== 'string' || !directory) {
    throw new TypeError('directory is required for durable owner messages');
  }
  if (!Number.isInteger(pageSize) || pageSize < 1) throw new RangeError('pageSize must be positive');
  if (!Number.isFinite(tombstoneRetentionMs) || tombstoneRetentionMs < 30 * 24 * 60 * 60 * 1000) {
    throw new RangeError('tombstoneRetentionMs must cover at least 30 days');
  }
  if (typeof now !== 'function') throw new TypeError('now must be a function');
  mkdirSync(directory, { recursive: true });
  const statePath = path.join(directory, 'owner-messages.json');
  let state = {
    nextSequence: 1,
    entries: [],
    lastSequenceByChat: {},
    unresolvedLidFences: {},
    tombstones: {},
  };
  try {
    const parsed = JSON.parse(readFileSync(statePath, 'utf8'));
    state = validateOwnerMessageState(parsed);
    for (const entry of state.entries) {
      const chat = entry.event.chatId;
      state.lastSequenceByChat[chat] = Math.max(
        Number(state.lastSequenceByChat[chat] || 0), entry.sequence,
      );
    }
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  const queuedIds = state.entries.map(entry => entry?.event?.messageId).filter(Boolean);
  const seen = new Set([...queuedIds, ...Object.keys(state.tombstones)]);

  function persist() {
    const temporaryPath = `${statePath}.${process.pid}.${randomBytes(6).toString('hex')}.tmp`;
    const descriptor = openSync(temporaryPath, 'w', 0o600);
    try {
      writeFileSync(descriptor, `${JSON.stringify(state)}\n`, { encoding: 'utf8' });
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
    renameSync(temporaryPath, statePath);
    let directoryDescriptor;
    try {
      directoryDescriptor = openSync(directory, 'r');
      fsyncSync(directoryDescriptor);
    } finally {
      if (directoryDescriptor !== undefined) closeSync(directoryDescriptor);
    }
  }

  function pruneExpiredTombstones(referenceTime = now()) {
    const cutoff = referenceTime - tombstoneRetentionMs;
    let changed = false;
    for (const [messageId, tombstone] of Object.entries(state.tombstones)) {
      if (!Number.isFinite(tombstone?.acknowledgedAt) || tombstone.acknowledgedAt <= cutoff) {
        delete state.tombstones[messageId];
        seen.delete(messageId);
        changed = true;
      }
    }
    return changed;
  }

  if (pruneExpiredTombstones()) persist();

  function mappedIdentifiers(chatId) {
    const identifier = normalizeWhatsAppIdentifier(chatId);
    if (!identifier) return [];
    return [...expandWhatsAppIdentifiers(identifier, directory)]
      .filter(alias => alias !== identifier);
  }

  function reconcileUnresolvedLidFences() {
    let changed = false;
    for (const [lidChatId, sequence] of Object.entries(state.unresolvedLidFences)) {
      const aliases = mappedIdentifiers(lidChatId);
      if (aliases.length === 0) continue;
      state.lastSequenceByChat[lidChatId] = Math.max(
        Number(state.lastSequenceByChat[lidChatId] || 0), sequence,
      );
      for (const alias of aliases) {
        const aliasChatId = `${alias}@s.whatsapp.net`;
        state.lastSequenceByChat[aliasChatId] = Math.max(
          Number(state.lastSequenceByChat[aliasChatId] || 0), sequence,
        );
      }
      delete state.unresolvedLidFences[lidChatId];
      changed = true;
    }
    // Persist exact alias keys and remove the conservative fence in the same
    // atomic replacement before any automatic send is released.
    if (changed) persist();
  }

  return {
    add(event) {
      const messageId = event?.messageId;
      const chatId = event?.chatId;
      const messageType = event?.type || (event?.hasMedia ? event?.mediaType : 'text');
      const timestamp = typeof event?.timestamp === 'number'
        ? event.timestamp
        : Number(event?.timestamp?.toString?.());
      if (event?.fromOwner !== true) return false;
      if (typeof messageId !== 'string' || !/^[A-Za-z0-9_-]{1,191}$/.test(messageId)) return false;
      if (typeof chatId !== 'string' || !/^\d{7,20}@(s\.whatsapp\.net|lid)$/.test(chatId)) return false;
      if (typeof event.senderId !== 'string' || !event.senderId) return false;
      if (typeof event.body !== 'string') return false;
      if (!Number.isFinite(timestamp) || timestamp <= 0) return false;
      const prunedTombstones = pruneExpiredTombstones();
      if (seen.has(messageId)) {
        if (prunedTombstones) persist();
        return false;
      }
      const queuedEvent = {
        messageId,
        chatId,
        senderId: event.senderId,
        fromOwner: true,
        type: messageType,
        body: event.body,
        timestamp,
        ownerFenceSequence: state.nextSequence,
      };
      if (messageType !== 'text') {
        queuedEvent.disposition = 'unsupported_media';
        queuedEvent.requiresIntervention = true;
      }
      seen.add(messageId);
      state.entries.push({ sequence: state.nextSequence, event: queuedEvent });
      state.lastSequenceByChat[chatId] = state.nextSequence;
      if (chatId.endsWith('@lid') && mappedIdentifiers(chatId).length === 0) {
        state.unresolvedLidFences[chatId] = state.nextSequence;
      }
      state.nextSequence += 1;
      persist();
      return true;
    },
    snapshot() {
      return state.entries.slice(0, pageSize).map(entry => entry.event);
    },
    page({ cursor = '0', limit = pageSize } = {}) {
      const afterSequence = Number(cursor || 0);
      const boundedLimit = Number.isInteger(limit) && limit > 0 ? Math.min(limit, pageSize) : pageSize;
      const available = state.entries.filter(entry => entry.sequence > afterSequence);
      const selected = available.slice(0, boundedLimit);
      return {
        messages: selected.map(entry => entry.event),
        nextCursor: selected.length ? String(selected[selected.length - 1].sequence) : String(afterSequence),
        hasMore: available.length > selected.length,
      };
    },
    acknowledge(acknowledgements) {
      const ids = new Set();
      for (const acknowledgement of acknowledgements || []) {
        const messageId = acknowledgement?.messageId;
        if (typeof messageId === 'string' && /^[A-Za-z0-9_-]{1,191}$/.test(messageId)) {
          ids.add(messageId);
        }
      }
      let removed = 0;
      const retained = [];
      const acknowledgedAt = now();
      for (const entry of state.entries) {
        if (ids.has(entry?.event?.messageId)) {
          state.tombstones[entry.event.messageId] = {
            sequence: entry.sequence,
            acknowledgedAt,
          };
          removed += 1;
        } else retained.push(entry);
      }
      if (removed) {
        state.entries = retained;
        persist();
      }
      return removed;
    },
    size() {
      return state.entries.length;
    },
    lastSequence(chatId) {
      reconcileUnresolvedLidFences();
      const exactChatId = String(chatId);
      const requestedIdentifier = normalizeWhatsAppIdentifier(exactChatId);
      let maximum = Number(state.lastSequenceByChat[exactChatId] || 0);
      if (!requestedIdentifier) return maximum;

      const aliases = expandWhatsAppIdentifiers(requestedIdentifier, directory);
      for (const [persistedChatId, sequence] of Object.entries(state.lastSequenceByChat)) {
        if (persistedChatId === exactChatId || !Number.isSafeInteger(sequence)) continue;
        const persistedIdentifier = normalizeWhatsAppIdentifier(persistedChatId);
        // The exact JID is always eligible. A different JID is eligible only
        // when an exact persisted session mapping reaches a different local
        // identifier; never infer that equal local parts across JID types are
        // aliases.
        if (
          persistedIdentifier
          && persistedIdentifier !== requestedIdentifier
          && aliases.has(persistedIdentifier)
        ) {
          maximum = Math.max(maximum, sequence);
        }
      }
      return maximum;
    },
    hasUnresolvedFence() {
      reconcileUnresolvedLidFences();
      return Object.keys(state.unresolvedLidFences).length > 0;
    },
  };
}

export function getMessageContent(msg) {
  const content = msg?.message || {};
  if (content.ephemeralMessage?.message) return content.ephemeralMessage.message;
  if (content.viewOnceMessage?.message) return content.viewOnceMessage.message;
  if (content.viewOnceMessageV2?.message) return content.viewOnceMessageV2.message;
  if (content.documentWithCaptionMessage?.message) return content.documentWithCaptionMessage.message;
  if (content.templateMessage?.hydratedTemplate) return content.templateMessage.hydratedTemplate;
  if (content.buttonsMessage) return content.buttonsMessage;
  if (content.listMessage) return content.listMessage;
  return content;
}

export function getContextInfo(messageContent) {
  if (!messageContent || typeof messageContent !== 'object') return {};
  for (const value of Object.values(messageContent)) {
    if (value && typeof value === 'object' && value.contextInfo) {
      return value.contextInfo;
    }
  }
  return {};
}

export function createBoundedMessageStore(limit = 512) {
  const byId = new Map();

  function remember(msg) {
    const id = msg?.key?.id;
    if (!id) return;
    byId.delete(id);
    byId.set(id, msg);
    while (byId.size > limit) {
      const oldest = byId.keys().next().value;
      byId.delete(oldest);
    }
  }

  function get(id) {
    if (!id || !byId.has(id)) return null;
    const msg = byId.get(id);
    byId.delete(id);
    byId.set(id, msg);
    return msg;
  }

  return { remember, get };
}

export function pollCreationMessageSecret(pollCreation) {
  return pollCreation?.message?.messageContextInfo?.messageSecret
    || pollCreation?.messageContextInfo?.messageSecret
    || null;
}

function uniqueStrings(values) {
  const seen = new Set();
  const out = [];
  for (const value of values || []) {
    const text = String(value || '').trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  return out;
}

export function pollUpdateForAggregation({
  pollUpdateMessage,
  pollUpdateMessageKey,
  pollCreation,
  decryptPollVote,
  getKeyAuthor,
  meId = 'me',
  pollCreatorJids = [],
  voterJids = [],
}) {
  if (!pollUpdateMessage) return null;
  const updateKey = pollUpdateMessage.pollUpdateMessageKey
    || pollUpdateMessageKey
    || pollUpdateMessage.key;
  if (!updateKey) return null;

  if (pollUpdateMessage.vote?.selectedOptions) {
    return {
      pollUpdateMessageKey: updateKey,
      vote: pollUpdateMessage.vote,
      senderTimestampMs: pollUpdateMessage.senderTimestampMs,
    };
  }

  const creationKey = pollUpdateMessage.pollCreationMessageKey;
  const secret = pollCreationMessageSecret(pollCreation);
  if (
    !creationKey?.id
    || !secret
    || !pollUpdateMessage.vote?.encPayload
    || !pollUpdateMessage.vote?.encIv
    || typeof decryptPollVote !== 'function'
    || typeof getKeyAuthor !== 'function'
  ) {
    return null;
  }

  // Baileys poll decryption keys include both creator and voter JIDs.  On
  // WhatsApp LID chats, the poll creator can be the linked-device LID even
  // when sock.user.id is the classic @s.whatsapp.net JID.  Try the exact
  // candidates the live bridge knows before falling back to the generic helper.
  const creatorCandidates = uniqueStrings([
    ...pollCreatorJids,
    getKeyAuthor(creationKey, meId),
  ]);
  const voterCandidates = uniqueStrings([
    ...voterJids,
    getKeyAuthor(updateKey, meId),
  ]);

  let lastError = null;
  for (const pollCreatorJid of creatorCandidates) {
    for (const voterJid of voterCandidates) {
      try {
        const vote = decryptPollVote(pollUpdateMessage.vote, {
          pollCreatorJid,
          pollMsgId: creationKey.id,
          pollEncKey: secret,
          voterJid,
        });
        return {
          pollUpdateMessageKey: updateKey,
          vote,
          senderTimestampMs: pollUpdateMessage.senderTimestampMs,
        };
      } catch (err) {
        lastError = err;
      }
    }
  }
  if (lastError) throw lastError;
  return null;
}

export function buildTextSendPayload(text, { replyTo, messageStore } = {}) {
  const content = { text };
  const options = {};
  const quoted = messageStore?.get(replyTo);
  if (quoted?.key && quoted?.message) {
    // Baileys expects quoted messages as sendMessage options, not inside the
    // message content payload. Keeping this split avoids silently sending a
    // literal/ignored `quoted` field instead of a native WhatsApp reply.
    options.quoted = quoted;
  }
  return { content, options };
}

export function buildLocationPayload({ latitude, longitude, name, address } = {}) {
  const lat = Number(latitude);
  const lon = Number(longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    throw new Error('latitude and longitude must be numbers');
  }
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
    throw new Error('latitude/longitude out of range');
  }

  const location = {
    degreesLatitude: lat,
    degreesLongitude: lon,
  };
  if (name) location.name = String(name);
  if (address) location.address = String(address);
  return { location };
}

function textFromQuotedMessage(quotedMessage) {
  if (!quotedMessage) return '';
  if (quotedMessage.conversation) return quotedMessage.conversation;
  if (quotedMessage.extendedTextMessage?.text) return quotedMessage.extendedTextMessage.text;
  if (quotedMessage.imageMessage?.caption) return quotedMessage.imageMessage.caption;
  if (quotedMessage.videoMessage?.caption) return quotedMessage.videoMessage.caption;
  if (quotedMessage.documentMessage?.caption) return quotedMessage.documentMessage.caption;
  if (quotedMessage.documentMessage?.fileName) return `[Document: ${quotedMessage.documentMessage.fileName}]`;
  if (quotedMessage.locationMessage) return formatLocationText(quotedMessage.locationMessage, false);
  if (quotedMessage.contactMessage) return formatContactText(quotedMessage.contactMessage);
  if (quotedMessage.pollCreationMessage) return formatPollText(quotedMessage.pollCreationMessage);
  return '';
}

function mediaExtForMime(mime, fallback) {
  const normalized = String(mime || '').split(';', 1)[0].toLowerCase();
  const extMap = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/webp': '.webp',
    'image/gif': '.gif',
    'video/mp4': '.mp4',
    'video/quicktime': '.mov',
    'video/x-matroska': '.mkv',
    'audio/ogg': '.ogg',
    'audio/mp4': '.m4a',
    'audio/mpeg': '.mp3',
    'application/pdf': '.pdf',
  };
  return extMap[normalized] || fallback;
}

function defaultWriteMediaFile({ buffer, dir, prefix, ext, fileName }) {
  mkdirSync(dir, { recursive: true });
  let safeName = fileName ? `_${path.basename(fileName).replace(/[^a-zA-Z0-9._-]/g, '_')}` : '';
  if (safeName && ext && !path.extname(safeName)) {
    safeName = `${safeName}${ext}`;
  }
  const filePath = path.join(dir, `${prefix}_${randomBytes(6).toString('hex')}${safeName || ext}`);
  writeFileSync(filePath, buffer);
  return filePath;
}

function formatLocationText(location, isLive) {
  const name = location.name || location.address || '';
  const lat = location.degreesLatitude ?? location.latitude;
  const lng = location.degreesLongitude ?? location.longitude;
  const kind = isLive ? 'Live location' : 'Location';
  const coords = lat !== undefined && lng !== undefined ? `${lat},${lng}` : '';
  return `[${kind}: ${[name, coords].filter(Boolean).join(' ')}]`;
}

function locationMetadata(location, isLive) {
  return {
    name: location.name || '',
    address: location.address || '',
    latitude: location.degreesLatitude ?? location.latitude ?? null,
    longitude: location.degreesLongitude ?? location.longitude ?? null,
    isLive,
  };
}

function formatContactText(contact) {
  const name = contact.displayName || contact.vcard?.match(/FN:(.+)/)?.[1] || 'unknown';
  const phone = contact.vcard?.match(/TEL[^:]*:(.+)/)?.[1] || '';
  return `[Contact: ${[name, phone].filter(Boolean).join(' ')}]`;
}

function formatContactsText(contacts) {
  const names = contacts.map(c => c.displayName).filter(Boolean);
  return `[Contacts: ${names.join(', ') || contacts.length}]`;
}

function formatReactionText(reaction) {
  const emoji = reaction.text || '';
  const target = reaction.key?.id || '';
  return `[Reaction: ${emoji}${target ? ` to ${target}` : ''}]`;
}

function pollOptions(poll) {
  return (poll.options || [])
    .map(option => option.optionName || option.name)
    .filter(Boolean);
}

function formatPollText(poll) {
  const question = poll.name || poll.title || 'poll';
  const options = pollOptions(poll);
  return `[Poll: ${question}${options.length ? ` Options: ${options.join(', ')}` : ''}]`;
}

function formatPollUpdateText(update) {
  const target = update.pollCreationMessageKey?.id || update.key?.id || '';
  return `[Poll update${target ? `: ${target}` : ''}]`;
}

/**
 * Append a visible note for media that failed to download, so the agent knows
 * something was sent rather than silently losing the attachment. Returns
 * `content` unchanged when nothing failed. (Port of nanoclaw#2895.)
 */
export function appendMediaFailureNote(content, failures) {
  if (!failures || failures.length === 0) return content;
  const note = failures.map((t) => `[${t} could not be downloaded]`).join(' ');
  return content ? `${content}\n${note}` : note;
}

export async function extractBridgeEvent({
  msg,
  chatId,
  senderId,
  senderNumber,
  botIds = [],
  isGroup = false,
  downloadMedia,
  writeMediaFile,
  cacheDirs = {},
}) {
  const messageContent = getMessageContent(msg);
  const contextInfo = getContextInfo(messageContent);
  const mentionedIds = Array.from(new Set((contextInfo?.mentionedJid || []).map(normalizeWhatsAppId).filter(Boolean)));
  const quotedMessageId = contextInfo?.stanzaId || null;
  const quotedParticipant = normalizeWhatsAppId(contextInfo?.participant || '') || null;
  const quotedRemoteJid = normalizeWhatsAppId(contextInfo?.remoteJid || '') || null;
  const hasQuotedMessage = !!contextInfo?.quotedMessage;
  const quotedText = textFromQuotedMessage(contextInfo?.quotedMessage);

  let body = '';
  let hasMedia = false;
  let mediaType = '';
  let mime = '';
  let fileName = '';
  let nativeType = '';
  const mediaUrls = [];
  const nativeMetadata = {};

  const mediaFailures = [];

  const saveMedia = async ({ mediaMessage, dir, prefix, fallbackExt, fileName: name, type }) => {
    if (!downloadMedia) return;
    try {
      const buf = await downloadMedia(msg);
      const ext = mediaExtForMime(mediaMessage?.mimetype, fallbackExt);
      const writer = writeMediaFile || defaultWriteMediaFile;
      const saved = await writer({ buffer: buf, dir, prefix, ext, fileName: name });
      if (saved) mediaUrls.push(saved);
    } catch (err) {
      // A failed CDN fetch (expired media URL, transient network error) must
      // never reject out of extractBridgeEvent — that would drop this message
      // AND every remaining message in the same upsert batch. Record the
      // failure so the agent is told media was sent instead of losing it
      // silently. (Port of nanoclaw#2895's never-silently-drop guarantee; the
      // reuploadRequest recovery half is already wired in bridge.js.)
      mediaFailures.push(type || 'media');
      try {
        console.warn(`[bridge] failed to download inbound ${type || 'media'}:`, err?.message || err);
      } catch {}
    }
  };

  if (messageContent.conversation) {
    body = messageContent.conversation;
    nativeType = 'conversation';
  } else if (messageContent.extendedTextMessage?.text) {
    body = messageContent.extendedTextMessage.text;
    nativeType = 'extendedTextMessage';
  } else if (messageContent.imageMessage) {
    const item = messageContent.imageMessage;
    body = item.caption || '';
    hasMedia = true;
    mediaType = 'image';
    nativeType = 'imageMessage';
    mime = item.mimetype || 'image/jpeg';
    await saveMedia({ mediaMessage: item, dir: cacheDirs.image, prefix: 'img', fallbackExt: '.jpg', type: 'image' });
  } else if (messageContent.videoMessage) {
    const item = messageContent.videoMessage;
    body = item.caption || '';
    hasMedia = true;
    mediaType = item.gifPlayback ? 'gif' : 'video';
    nativeType = 'videoMessage';
    mime = item.mimetype || 'video/mp4';
    nativeMetadata.video = { gifPlayback: !!item.gifPlayback };
    await saveMedia({ mediaMessage: item, dir: cacheDirs.document, prefix: 'vid', fallbackExt: '.mp4', type: mediaType });
  } else if (messageContent.audioMessage || messageContent.pttMessage) {
    const item = messageContent.pttMessage || messageContent.audioMessage;
    hasMedia = true;
    mediaType = item.ptt || messageContent.pttMessage ? 'ptt' : 'audio';
    nativeType = messageContent.pttMessage ? 'pttMessage' : 'audioMessage';
    mime = item.mimetype || 'audio/ogg';
    nativeMetadata.audio = { ptt: mediaType === 'ptt' };
    await saveMedia({ mediaMessage: item, dir: cacheDirs.audio, prefix: 'aud', fallbackExt: '.ogg', type: 'audio' });
  } else if (messageContent.documentMessage) {
    const item = messageContent.documentMessage;
    body = item.caption || '';
    hasMedia = true;
    mediaType = 'document';
    nativeType = 'documentMessage';
    mime = item.mimetype || 'application/octet-stream';
    fileName = item.fileName || 'document';
    await saveMedia({ mediaMessage: item, dir: cacheDirs.document, prefix: 'doc', fallbackExt: '.bin', fileName, type: 'document' });
  } else if (messageContent.stickerMessage) {
    hasMedia = true;
    mediaType = 'sticker';
    nativeType = 'stickerMessage';
    mime = messageContent.stickerMessage.mimetype || 'image/webp';
    body = '[Sticker]';
    nativeMetadata.sticker = {
      animated: !!messageContent.stickerMessage.isAnimated,
      mimetype: mime,
    };
    await saveMedia({ mediaMessage: messageContent.stickerMessage, dir: cacheDirs.image, prefix: 'sticker', fallbackExt: '.webp', type: 'sticker' });
  } else if (messageContent.locationMessage || messageContent.liveLocationMessage) {
    const isLive = !!messageContent.liveLocationMessage;
    const item = messageContent.liveLocationMessage || messageContent.locationMessage;
    mediaType = isLive ? 'live_location' : 'location';
    nativeType = isLive ? 'liveLocationMessage' : 'locationMessage';
    body = formatLocationText(item, isLive);
    nativeMetadata.location = locationMetadata(item, isLive);
  } else if (messageContent.contactMessage) {
    mediaType = 'contact';
    nativeType = 'contactMessage';
    body = formatContactText(messageContent.contactMessage);
    nativeMetadata.contact = {
      displayName: messageContent.contactMessage.displayName || '',
      vcard: messageContent.contactMessage.vcard || '',
    };
  } else if (messageContent.contactsArrayMessage) {
    const contacts = messageContent.contactsArrayMessage.contacts || [];
    mediaType = 'contacts';
    nativeType = 'contactsArrayMessage';
    body = formatContactsText(contacts);
    nativeMetadata.contacts = contacts.map(contact => ({
      displayName: contact.displayName || '',
      vcard: contact.vcard || '',
    }));
  } else if (messageContent.reactionMessage) {
    mediaType = 'reaction';
    nativeType = 'reactionMessage';
    body = formatReactionText(messageContent.reactionMessage);
    nativeMetadata.reaction = {
      text: messageContent.reactionMessage.text || '',
      messageId: messageContent.reactionMessage.key?.id || '',
      remoteJid: normalizeWhatsAppId(messageContent.reactionMessage.key?.remoteJid || ''),
      participant: normalizeWhatsAppId(messageContent.reactionMessage.key?.participant || ''),
    };
  } else if (messageContent.pollCreationMessage || messageContent.pollCreationMessageV2 || messageContent.pollCreationMessageV3) {
    const item = messageContent.pollCreationMessage || messageContent.pollCreationMessageV2 || messageContent.pollCreationMessageV3;
    mediaType = 'poll';
    nativeType = messageContent.pollCreationMessage ? 'pollCreationMessage' : messageContent.pollCreationMessageV2 ? 'pollCreationMessageV2' : 'pollCreationMessageV3';
    body = formatPollText(item);
    nativeMetadata.poll = {
      question: item.name || item.title || '',
      options: pollOptions(item),
      selectableCount: item.selectableOptionsCount || item.selectableCount || 1,
    };
  } else if (messageContent.pollUpdateMessage) {
    mediaType = 'poll_update';
    nativeType = 'pollUpdateMessage';
    body = formatPollUpdateText(messageContent.pollUpdateMessage);
    nativeMetadata.pollUpdate = messageContent.pollUpdateMessage;
  }

  // Surface failed downloads to the agent instead of silently losing the
  // attachment. Applied before the generic "[<type> received]" fallback so an
  // uncaptioned message whose download failed reads "[image could not be
  // downloaded]" rather than claiming the media arrived.
  body = appendMediaFailureNote(body, mediaFailures);

  if (hasMedia && !body) {
    body = `[${mediaType} received]`;
  }

  return {
    messageId: msg.key.id,
    chatId,
    senderId,
    senderName: msg.pushName || senderNumber,
    chatName: isGroup ? (chatId.split('@')[0]) : (msg.pushName || senderNumber),
    isGroup,
    body,
    hasMedia,
    mediaType,
    mime,
    fileName,
    nativeType,
    nativeMetadata,
    mediaUrls,
    mentionedIds,
    quotedMessageId,
    quotedParticipant,
    quotedRemoteJid,
    quotedText,
    hasQuotedMessage,
    botIds,
    readReceiptKey: {
      remoteJid: msg.key.remoteJid || chatId,
      id: msg.key.id,
      participant: msg.key.participant || senderId,
      fromMe: Boolean(msg.key.fromMe),
    },
    timestamp: msg.messageTimestamp,
  };
}

export function inferMediaType(ext) {
  if (['jpg', 'jpeg', 'png', 'webp', 'gif'].includes(ext)) return 'image';
  if (['mp4', 'mov', 'avi', 'mkv', '3gp'].includes(ext)) return 'video';
  if (['ogg', 'opus', 'mp3', 'wav', 'm4a'].includes(ext)) return 'audio';
  return 'document';
}

export function inboundReadReceiptKeys({ key, enabled }) {
  if (!enabled || !key || key.fromMe || !key.id || !key.remoteJid) return [];
  // Preserve participant for group messages: Baileys needs the original key.
  return [key];
}

export function mediaPayloadForFile({ buffer, filePath, mediaType, caption, fileName }) {
  const ext = filePath.toLowerCase().split('.').pop();
  const type = mediaType || inferMediaType(ext);
  if (type === 'image' && ext === 'gif') {
    // Pure helper fallback: do not lie and label raw GIF bytes as mp4.
    // The live bridge tries ffmpeg conversion to WhatsApp gifPlayback video
    // before it falls back to this regular image payload.
    return { image: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'image/gif' };
  }
  switch (type) {
    case 'image':
      return { image: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'image/jpeg' };
    case 'video':
      return { video: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'video/mp4' };
    case 'document':
      return {
        document: buffer,
        fileName: fileName || path.basename(filePath),
        caption: caption || undefined,
        mimetype: MIME_MAP[ext] || 'application/octet-stream',
      };
    default:
      return null;
  }
}

export function buildPollPayload({ question, options, selectableCount = 1 }) {
  const cleanQuestion = String(question || '').trim();
  const cleanOptions = (options || []).map(option => String(option || '').trim()).filter(Boolean);
  if (!cleanQuestion) throw new Error('question is required');
  if (cleanOptions.length < 2) throw new Error('at least two poll options are required');
  if (cleanOptions.length > 12) throw new Error('at most 12 poll options are supported');
  const count = Math.max(1, Math.min(Number(selectableCount) || 1, cleanOptions.length));
  return {
    poll: {
      name: cleanQuestion,
      values: cleanOptions,
      selectableCount: count,
      messageSecret: randomBytes(32),
    },
  };
}

export function pollCreationMessageFromPayload(payload) {
  const poll = payload?.poll;
  if (!poll) return null;
  const values = Array.isArray(poll.values) ? poll.values : [];
  const options = values.map(value => String(value || '').trim()).filter(Boolean);
  if (!poll.name || options.length < 2) return null;
  const selectableOptionsCount = Math.max(1, Math.min(Number(poll.selectableCount) || 1, options.length));
  const message = {};
  if (poll.messageSecret) {
    message.messageContextInfo = { messageSecret: poll.messageSecret };
  }
  message[selectableOptionsCount === 1 ? 'pollCreationMessageV3' : 'pollCreationMessage'] = {
    name: String(poll.name),
    options: options.map(optionName => ({ optionName })),
    selectableOptionsCount,
  };
  return message;
}

/**
 * Reconnect scheduling guard. startSocket() awaits network I/O before it
 * creates a socket or registers event handlers, so a bare
 * `setTimeout(startSocket, ...)` has two unrecoverable failure modes: a
 * rejection is unhandled (crashes the process on modern Node), and a hang
 * leaves the bridge permanently disconnected with nothing left to retry.
 * Every (re)connect must go through the scheduler this returns.
 */
export function createReconnectScheduler(startFn, {
  retryDelayMs = 5000,
  log = console.log,
  setTimeoutFn = setTimeout,
} = {}) {
  function scheduleReconnect(delayMs) {
    setTimeoutFn(() => {
      Promise.resolve()
        .then(startFn)
        .catch((err) => {
          log(`⚠️  Reconnect failed (${err?.message || err}). Retrying in ${Math.round(retryDelayMs / 1000)}s...`);
          scheduleReconnect(retryDelayMs);
        });
    }, delayMs);
  }
  return scheduleReconnect;
}

/**
 * Version resolution guard. fetchLatestBaileysVersion() is a plain fetch to
 * raw.githubusercontent.com with no AbortSignal; a stalled connection can
 * pend forever and wedge the reconnect path (the scheduler above cannot
 * retry past an await that never settles). Bound the fetch and fall back to
 * the last known-good version, or the Baileys default before first success.
 */
export function createVersionResolver(fetchVersionFn, {
  timeoutMs = 15000,
  log = console.log,
} = {}) {
  let cachedVersion = null;
  return async function resolveVersion() {
    let timer = null;
    try {
      const { version } = await Promise.race([
        fetchVersionFn(),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error('version fetch timed out')), timeoutMs);
        }),
      ]);
      cachedVersion = version;
    } catch (err) {
      log(`⚠️  Baileys version fetch failed (${err?.message || err}); using ${cachedVersion ? 'cached version' : 'library default'}.`);
    } finally {
      if (timer) clearTimeout(timer);
    }
    return cachedVersion;
  };
}
