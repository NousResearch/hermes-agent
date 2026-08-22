import { createHash, timingSafeEqual } from 'crypto';
import {
  chmodSync,
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
  fsyncSync,
} from 'fs';
import path from 'path';

const CONTROL = /[\x00-\x1f\x7f]/;
const JID = /^(\d{1,32})(?::\d{1,3})?@(s\.whatsapp\.net|c\.us|lid)$/i;
const OPERATION_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{7,63}$/;
const SCHEMA_VERSION = 1;
const MAX_OPERATIONS = 256;

export function normalizeParticipantJid(value) {
  const raw = String(value || '').trim();
  if (!raw || raw.length > 96 || CONTROL.test(raw)) return null;
  const match = raw.match(JID);
  if (!match) return null;
  const domain = match[2].toLowerCase() === 'c.us' ? 's.whatsapp.net' : match[2].toLowerCase();
  return `${match[1]}@${domain}`;
}

export function parseParticipantAllowlist(raw) {
  const result = new Set();
  for (const value of String(raw || '').split(',')) {
    if (!value.trim()) continue;
    const normalized = normalizeParticipantJid(value);
    if (!normalized) return new Set();
    result.add(normalized);
  }
  return result;
}

export function authorizeGroupControl(header, expectedToken) {
  const prefix = 'Bearer ';
  const supplied = String(header || '');
  const expected = String(expectedToken || '');
  if (expected.length < 32 || expected.length > 512 || !supplied.startsWith(prefix)) return false;
  const actual = supplied.slice(prefix.length);
  const actualBuffer = Buffer.from(actual, 'utf8');
  const expectedBuffer = Buffer.from(expected, 'utf8');
  if (actualBuffer.length !== expectedBuffer.length) return false;
  return timingSafeEqual(actualBuffer, expectedBuffer);
}

export function validateGroupCreatePayload(body, allowedParticipants) {
  const subject = String(body?.subject || '').trim();
  const confirmedSubject = String(body?.confirmedSubject || '');
  const operationId = String(body?.operationId || '').trim();
  if (
    !subject || subject.length > 100 || CONTROL.test(subject)
    || confirmedSubject !== subject
    || !OPERATION_ID.test(operationId)
    || !Array.isArray(body?.participants)
    || !Array.isArray(body?.confirmedParticipants)
    || body.participants.length < 1
    || body.participants.length > 50
    || !(allowedParticipants instanceof Set)
    || allowedParticipants.size < 1
  ) return null;

  const participants = [];
  for (const raw of body.participants) {
    const participant = normalizeParticipantJid(raw);
    if (!participant || !allowedParticipants.has(participant) || participants.includes(participant)) {
      return null;
    }
    participants.push(participant);
  }
  const confirmedParticipants = body.confirmedParticipants.map(normalizeParticipantJid);
  if (
    confirmedParticipants.length !== participants.length
    || confirmedParticipants.some((participant, index) => participant !== participants[index])
  ) return null;
  const payloadHash = createHash('sha256')
    .update(JSON.stringify({ subject, participants }))
    .digest('hex');
  return { subject, confirmedSubject, operationId, participants, confirmedParticipants, payloadHash };
}

function validateState(value) {
  if (!value || value.schemaVersion !== SCHEMA_VERSION || !Array.isArray(value.operations)) {
    throw new Error('Group operation state is invalid');
  }
  for (const operation of value.operations) {
    if (
      !operation || !OPERATION_ID.test(String(operation.operationId || ''))
      || !/^[a-f0-9]{64}$/.test(String(operation.payloadHash || ''))
      || !['pending', 'created', 'uncertain'].includes(operation.status)
      || typeof operation.subject !== 'string' || !operation.subject
      || !Number.isFinite(operation.createdAt) || !Number.isFinite(operation.updatedAt)
      || (operation.groupId !== undefined && !String(operation.groupId).endsWith('@g.us'))
    ) throw new Error('Group operation state is invalid');
  }
  return value;
}

export class GroupOperationStore {
  constructor(filePath, { now = () => Date.now() } = {}) {
    if (!filePath) throw new Error('Group operation state path is required');
    this.filePath = filePath;
    this.now = now;
  }

  load() {
    if (!existsSync(this.filePath)) return { schemaVersion: SCHEMA_VERSION, operations: [] };
    return validateState(JSON.parse(readFileSync(this.filePath, 'utf8')));
  }

  save(state) {
    validateState(state);
    const directory = path.dirname(this.filePath);
    mkdirSync(directory, { recursive: true, mode: 0o700 });
    chmodSync(directory, 0o700);
    const temporary = `${this.filePath}.${process.pid}.tmp`;
    let descriptor;
    try {
      descriptor = openSync(temporary, 'wx', 0o600);
      writeFileSync(descriptor, `${JSON.stringify(state, null, 2)}\n`);
      fsyncSync(descriptor);
      closeSync(descriptor);
      descriptor = undefined;
      renameSync(temporary, this.filePath);
      chmodSync(this.filePath, 0o600);
      const directoryDescriptor = openSync(directory, 'r');
      try { fsyncSync(directoryDescriptor); } finally { closeSync(directoryDescriptor); }
    } finally {
      if (descriptor !== undefined) closeSync(descriptor);
      if (existsSync(temporary)) unlinkSync(temporary);
    }
  }

  recordPending(request) {
    const state = this.load();
    const existing = state.operations.find(item => item.operationId === request.operationId);
    if (existing) return { existing };
    const reserved = state.operations.find(
      item => item.subject === request.subject || item.payloadHash === request.payloadHash,
    );
    if (reserved) return { reserved };
    const now = this.now();
    const recentAttempt = state.operations.find(
      item => Number.isFinite(item.createdAt) && now - item.createdAt < request.minimumIntervalMs,
    );
    if (recentAttempt) return { rateLimited: true };
    const operation = {
      operationId: request.operationId,
      payloadHash: request.payloadHash,
      subject: request.subject,
      status: 'pending',
      createdAt: now,
      updatedAt: now,
    };
    if (state.operations.length >= MAX_OPERATIONS) {
      throw new Error('Group operation state is at capacity');
    }
    state.operations.push(operation);
    this.save(state);
    return { operation };
  }

  finish(operationId, status, groupId = null) {
    const state = this.load();
    const operation = state.operations.find(item => item.operationId === operationId);
    if (!operation) throw new Error('Group operation is missing');
    operation.status = status;
    operation.updatedAt = this.now();
    if (groupId) operation.groupId = String(groupId).slice(0, 128);
    this.save(state);
    return operation;
  }
}

function existingOperationResponse(existing, request) {
  if (existing.payloadHash !== request.payloadHash) {
    return { httpStatus: 409, body: { success: false, status: 'conflict', error: 'Operation ID is already bound to a different request.' } };
  }
  if (existing.status === 'created' && existing.groupId) {
    return { httpStatus: 200, body: { success: true, status: 'created', groupId: existing.groupId } };
  }
  return { httpStatus: 409, body: { success: false, status: 'uncertain', error: 'This operation is pending or uncertain and will not be retried automatically.' } };
}

function reservedOperationResponse() {
  return { httpStatus: 409, body: { success: false, status: 'reserved', error: 'This group subject or participant payload is already bound to another operation and will not be retried.' } };
}

function withTimeout(promise, timeoutMs) {
  let timer;
  return Promise.race([
    Promise.resolve(promise),
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('Group control timed out')), timeoutMs);
    }),
  ]).finally(() => clearTimeout(timer));
}

export async function executeGroupCreate({
  body,
  allowedParticipants,
  store,
  listGroups,
  createGroup,
  now = () => Date.now(),
  minimumIntervalMs = 30_000,
  operationTimeoutMs = 10_000,
}) {
  const request = validateGroupCreatePayload(body, allowedParticipants);
  if (!request) {
    return { httpStatus: 400, body: { success: false, status: 'invalid', error: 'Group request is invalid.' } };
  }

  const state = store.load();
  const existing = state.operations.find(item => item.operationId === request.operationId);
  if (existing) return existingOperationResponse(existing, request);
  const reservedOperation = state.operations.find(
    item => item.subject === request.subject || item.payloadHash === request.payloadHash,
  );
  if (reservedOperation) return reservedOperationResponse();
  const recentAttempt = state.operations.find(
    item => Number.isFinite(item.createdAt) && now() - item.createdAt < minimumIntervalMs,
  );
  if (recentAttempt) {
    return { httpStatus: 429, body: { success: false, status: 'rate-limited', error: 'Group creation is temporarily rate limited.' } };
  }

  let groups;
  try {
    groups = await withTimeout(listGroups(), operationTimeoutMs);
  } catch {
    return { httpStatus: 503, body: { success: false, status: 'unavailable', error: 'Existing groups could not be verified.' } };
  }
  if (Object.values(groups || {}).some(group => group?.subject === request.subject)) {
    return { httpStatus: 409, body: { success: false, status: 'subject-exists', error: 'A group with this exact subject already exists.' } };
  }

  const reserved = store.recordPending({ ...request, minimumIntervalMs });
  if (reserved.existing) return existingOperationResponse(reserved.existing, request);
  if (reserved.reserved) return reservedOperationResponse();
  if (reserved.rateLimited) {
    return { httpStatus: 429, body: { success: false, status: 'rate-limited', error: 'Group creation is temporarily rate limited.' } };
  }
  try {
    const created = await withTimeout(
      createGroup(request.subject, request.participants),
      operationTimeoutMs,
    );
    const groupId = String(created?.id || '').trim();
    if (!groupId || !groupId.endsWith('@g.us')) throw new Error('Group result is invalid');
    store.finish(request.operationId, 'created', groupId);
    return { httpStatus: 201, body: { success: true, status: 'created', groupId } };
  } catch {
    store.finish(request.operationId, 'uncertain');
    return { httpStatus: 502, body: { success: false, status: 'uncertain', error: 'Group creation result is uncertain and will not be retried automatically.' } };
  }
}
