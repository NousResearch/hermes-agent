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
    broadcasts: {
      channels: {},
    },
    kns_cache: {
      by_name: {},
      by_address: {},
    },
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
    last_live_tx_seen_ms: Number(existing.last_live_tx_seen_ms || 0),
    pending_handshake: existing.pending_handshake || null,
    pending_outbound_handshake: existing.pending_outbound_handshake || null,
    nickname:
      String(existing.nickname || "").trim() || null,
    kns_name:
      String(existing.kns_name || "").trim() || null,
    display_name:
      String(existing.display_name || "").trim() || null,
    identity_source:
      String(existing.identity_source || "").trim() || null,
    last_identity_refresh_ms: Number(existing.last_identity_refresh_ms || 0),
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
  const indexedTxIds = Array.isArray(existing.indexed_tx_ids || existing.indexedTxIds)
    ? (existing.indexed_tx_ids || existing.indexedTxIds).filter(
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
    submitted_ms:
      existing.submitted_ms == null && existing.submittedMs == null
        ? null
        : Number(existing.submitted_ms ?? existing.submittedMs),
    observed_live_ms:
      existing.observed_live_ms == null && existing.observedLiveMs == null
        ? null
        : Number(existing.observed_live_ms ?? existing.observedLiveMs),
    indexed_ms:
      existing.indexed_ms == null && existing.indexedMs == null
        ? null
        : Number(existing.indexed_ms ?? existing.indexedMs),
    indexed_block_time_ms:
      existing.indexed_block_time_ms == null &&
      existing.indexedBlockTimeMs == null
        ? null
        : Number(
            existing.indexed_block_time_ms ?? existing.indexedBlockTimeMs
          ),
    total_parts: Number(existing.total_parts || existing.totalParts || 0),
    completed_parts: Number(
      existing.completed_parts || existing.completedParts || 0
    ),
    indexed_parts: Number(existing.indexed_parts || existing.indexedParts || 0),
    tx_ids: txIds,
    indexed_tx_ids: indexedTxIds,
    last_tx_id:
      String(existing.last_tx_id || existing.lastTxId || "").trim() || null,
    error: existing.error == null ? null : String(existing.error),
    message_preview:
      String(existing.message_preview || existing.messagePreview || "").trim() ||
      null,
    job_kind:
      String(existing.job_kind || existing.jobKind || "").trim() || "dm",
  };
}

function normalizeKnsEntry(entry = {}, fallbackKey = "") {
  const cacheKey = String(entry.key || fallbackKey || "").trim().toLowerCase();
  if (!cacheKey) {
    return null;
  }
  const normalized = {
    key: cacheKey,
    resolved_at_ms: Number(entry.resolved_at_ms || entry.resolvedAtMs || 0),
    expires_at_ms: Number(entry.expires_at_ms || entry.expiresAtMs || 0),
  };
  if (entry.address != null) {
    normalized.address = String(entry.address || "").trim() || null;
  }
  if (entry.name != null) {
    normalized.name = String(entry.name || "").trim().toLowerCase() || null;
  }
  if (entry.error != null) {
    normalized.error = String(entry.error || "").trim() || null;
  }
  return normalized;
}

function normalizeBroadcastChannel(channelName, existing = {}) {
  const normalizedName = String(
    existing.channel_name || existing.channelName || channelName || ""
  )
    .trim()
    .toLowerCase();
  if (!normalizedName) {
    return null;
  }

  const publishers = Array.isArray(existing.publishers)
    ? existing.publishers
        .map((value) => String(value || "").trim())
        .filter(Boolean)
    : [];
  const recentMessages = Array.isArray(existing.recent_messages || existing.recentMessages)
    ? (existing.recent_messages || existing.recentMessages)
        .map((message) => ({
          tx_id: String(message?.tx_id || message?.txId || "").trim() || null,
          sender_address:
            String(message?.sender_address || message?.senderAddress || "").trim() ||
            null,
          content: String(message?.content || "").trim() || null,
          observed_live_ms:
            message?.observed_live_ms == null && message?.observedLiveMs == null
              ? null
              : Number(message?.observed_live_ms ?? message?.observedLiveMs),
          block_time_ms:
            message?.block_time_ms == null && message?.blockTimeMs == null
              ? null
              : Number(message?.block_time_ms ?? message?.blockTimeMs),
        }))
        .filter((message) => message.tx_id)
    : [];

  return {
    channel_name: normalizedName,
    channel_id:
      String(existing.channel_id || existing.channelId || "").trim() ||
      `broadcast:${normalizedName}`,
    publishers,
    allow_publish: Boolean(
      existing.allow_publish ?? existing.allowPublish ?? false
    ),
    updated_at: existing.updated_at || nowIso(),
    last_seen_block_time: Number(existing.last_seen_block_time || 0),
    last_live_poll_ms: Number(existing.last_live_poll_ms || 0),
    recent_messages: recentMessages.slice(-25),
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
    broadcasts: {
      channels: {},
      ...(state?.broadcasts || {}),
      channels: {},
    },
    kns_cache: {
      by_name: {},
      by_address: {},
      ...(state?.kns_cache || {}),
      by_name: {},
      by_address: {},
    },
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

  for (const [channelName, channel] of Object.entries(
    state?.broadcasts?.channels || {}
  )) {
    const normalizedChannel = normalizeBroadcastChannel(channelName, channel);
    if (normalizedChannel) {
      normalized.broadcasts.channels[normalizedChannel.channel_name] =
        normalizedChannel;
    }
  }

  for (const [cacheKey, entry] of Object.entries(
    state?.kns_cache?.by_name || {}
  )) {
    const normalizedEntry = normalizeKnsEntry(entry, cacheKey);
    if (normalizedEntry) {
      normalized.kns_cache.by_name[normalizedEntry.key] = normalizedEntry;
    }
  }

  for (const [cacheKey, entry] of Object.entries(
    state?.kns_cache?.by_address || {}
  )) {
    const normalizedEntry = normalizeKnsEntry(entry, cacheKey);
    if (normalizedEntry) {
      normalized.kns_cache.by_address[normalizedEntry.key] = normalizedEntry;
    }
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
  const tempPath = `${statePath}.${process.pid}.${Date.now()}.${Math.random()
    .toString(16)
    .slice(2)}.tmp`;
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

export function ensureBroadcastChannel(state, channelName, existing = {}) {
  const normalized = normalizeBroadcastChannel(channelName, existing);
  if (!normalized) {
    throw new Error("Broadcast channel name is required");
  }
  const current = state.broadcasts.channels[normalized.channel_name];
  state.broadcasts.channels[normalized.channel_name] = current
    ? normalizeBroadcastChannel(channelName, {
        ...current,
        channel_id: normalized.channel_id || current.channel_id,
        publishers: normalized.publishers,
        allow_publish: normalized.allow_publish,
      })
    : normalized;
  return state.broadcasts.channels[normalized.channel_name];
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

export function touchBroadcastChannel(channel, timestamp = nowIso()) {
  channel.updated_at = timestamp;
}
