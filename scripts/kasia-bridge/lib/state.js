import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

const DEFAULT_MAX_PROCESSED_TX_IDS = 1000;

function nowIso() {
  return new Date().toISOString();
}

export function createEmptyState(wallet = {}) {
  return {
    wallet: {
      address: wallet.address || null,
      public_key: wallet.public_key || null,
      network: wallet.network || null,
      send_state: wallet.send_state || {},
    },
    send_jobs: {},
    conversations: {},
    cursors: {
      handshakes_block_time: 0,
    },
    processed_tx_ids: [],
    last_sync_ms: null,
  };
}

function normalizeConversation(peerAddress, existing = {}) {
  const normalizedPeer = String(peerAddress || "").trim();
  return {
    conversation_id:
      existing.conversation_id || `kasia:${normalizedPeer}`,
    peer_address: normalizedPeer,
    our_alias: existing.our_alias || null,
    their_alias: existing.their_alias || null,
    status: existing.status || "pending",
    updated_at: existing.updated_at || nowIso(),
    last_handshake_block_time: Number(existing.last_handshake_block_time || 0),
    last_context_block_time: Number(existing.last_context_block_time || 0),
    pending_handshake: existing.pending_handshake || null,
  };
}

function normalizeSendJob(existing = {}) {
  const jobId = String(existing.job_id || existing.jobId || "").trim();
  if (!jobId) {
    return null;
  }

  const txIds = Array.isArray(existing.tx_ids || existing.txIds)
    ? (existing.tx_ids || existing.txIds).filter(
        (value) => typeof value === "string" && value.trim()
      )
    : [];

  return {
    job_id: jobId,
    chat_id: String(existing.chat_id || existing.chatId || "").trim() || null,
    status: String(existing.status || "queued").trim() || "queued",
    created_ms: Number(existing.created_ms || existing.createdMs || 0),
    updated_ms: Number(existing.updated_ms || existing.updatedMs || 0),
    started_ms:
      existing.started_ms == null && existing.startedMs == null
        ? null
        : Number(existing.started_ms ?? existing.startedMs),
    finished_ms:
      existing.finished_ms == null && existing.finishedMs == null
        ? null
        : Number(existing.finished_ms ?? existing.finishedMs),
    total_parts: Number(existing.total_parts || existing.totalParts || 0),
    completed_parts: Number(
      existing.completed_parts || existing.completedParts || 0
    ),
    tx_ids: txIds,
    last_tx_id:
      String(existing.last_tx_id || existing.lastTxId || "").trim() || null,
    error: existing.error == null ? null : String(existing.error),
    message_preview:
      String(existing.message_preview || existing.messagePreview || "").trim() ||
      null,
  };
}

function normalizeState(state, wallet = {}) {
  const base = createEmptyState(wallet);
  const normalized = {
    ...base,
    ...state,
    wallet: {
      ...base.wallet,
      ...(state?.wallet || {}),
      ...wallet,
      send_state: {
        ...(base.wallet.send_state || {}),
        ...((state?.wallet || {}).send_state || {}),
        ...(wallet.send_state || {}),
      },
    },
    conversations: {},
    cursors: {
      ...base.cursors,
      ...(state?.cursors || {}),
    },
    send_jobs: {},
    processed_tx_ids: Array.isArray(state?.processed_tx_ids)
      ? state.processed_tx_ids.filter((value) => typeof value === "string")
      : [],
    last_sync_ms:
      state?.last_sync_ms == null ? null : Number(state.last_sync_ms),
  };

  for (const [peerAddress, conversation] of Object.entries(
    state?.conversations || {}
  )) {
    const normalizedConversation = normalizeConversation(peerAddress, conversation);
    normalized.conversations[normalizedConversation.peer_address] =
      normalizedConversation;
  }

  for (const [jobId, job] of Object.entries(state?.send_jobs || {})) {
    const normalizedJob = normalizeSendJob({
      job_id: jobId,
      ...(job || {}),
    });
    if (normalizedJob) {
      normalized.send_jobs[normalizedJob.job_id] = normalizedJob;
    }
  }

  return normalized;
}

export async function loadState(statePath, wallet = {}) {
  try {
    const raw = await readFile(statePath, "utf8");
    return normalizeState(JSON.parse(raw), wallet);
  } catch (error) {
    if (error && error.code !== "ENOENT") {
      throw error;
    }
    return createEmptyState(wallet);
  }
}

export async function saveState(statePath, state) {
  await mkdir(dirname(statePath), { recursive: true });
  const tempPath = `${statePath}.tmp`;
  const payload = `${JSON.stringify(state, null, 2)}\n`;
  await writeFile(tempPath, payload, "utf8");
  await rename(tempPath, statePath);
}

export function ensureConversation(state, peerAddress) {
  const normalizedPeer = String(peerAddress || "").trim();
  if (!normalizedPeer) {
    throw new Error("Peer address is required");
  }
  if (!state.conversations[normalizedPeer]) {
    state.conversations[normalizedPeer] = normalizeConversation(normalizedPeer);
  }
  return state.conversations[normalizedPeer];
}

export function hasProcessedTx(state, txId) {
  return Boolean(txId) && state.processed_tx_ids.includes(txId);
}

export function markProcessedTx(
  state,
  txId,
  maxSize = DEFAULT_MAX_PROCESSED_TX_IDS
) {
  if (!txId || hasProcessedTx(state, txId)) {
    return;
  }
  state.processed_tx_ids.push(txId);
  if (state.processed_tx_ids.length > maxSize) {
    state.processed_tx_ids.splice(
      0,
      state.processed_tx_ids.length - maxSize
    );
  }
}

export function touchConversation(conversation, timestamp = nowIso()) {
  conversation.updated_at = timestamp;
}
