import { randomUUID } from "node:crypto";
import { join } from "node:path";

import {
  buildContextualMessageTransactionPayload,
  buildHandshakePayload,
  buildHandshakeTransactionPayload,
  decodeIndexedContextualMessagePayload,
  decryptSealedMessage,
  generateAlias,
  MINIMUM_MESSAGE_AMOUNT_SOMPI,
  parseHandshakePayload,
  shortenAddress,
} from "./protocol.js";
import {
  createEmptyState,
  ensureConversation,
  hasProcessedTx,
  loadState,
  markProcessedTx,
  saveState,
  touchConversation,
} from "./state.js";
import { KaspaWalletClient } from "./kaspa_wallet.js";

function joinUrl(baseUrl, relativePath, params = {}) {
  const normalizedBase = String(baseUrl || "").replace(/\/+$/, "");
  const normalizedPath = String(relativePath || "").replace(/^\/+/, "");
  const url = new URL(`${normalizedBase}/${normalizedPath}`);
  Object.entries(params).forEach(([key, value]) => {
    if (value != null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  return url;
}

function encodeIndexerAlias(alias) {
  return Buffer.from(String(alias || ""), "utf8").toString("hex");
}

const DEFAULT_CONTEXTUAL_MESSAGE_TARGET_CHARS = 240;
const DEFAULT_CONTEXTUAL_MESSAGE_MIN_CHARS = 40;
const DEFAULT_CONTEXTUAL_MESSAGE_MAX_PARTS = 8;
const DEFAULT_MAX_SEND_JOBS = 100;
const DEFAULT_SEND_JOB_PREVIEW_CHARS = 120;

function isTerminalSendJobStatus(status) {
  return status === "sent" || status === "failed" || status === "rejected";
}

function buildSendJobPreview(message, maxChars = DEFAULT_SEND_JOB_PREVIEW_CHARS) {
  const normalized = String(message || "").trim().replace(/\s+/g, " ");
  if (!normalized) {
    return null;
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(1, maxChars - 1)).trimEnd()}…`;
}

function toPublicSendJob(job) {
  if (!job) {
    return null;
  }
  return {
    jobId: job.job_id,
    chatId: job.chat_id,
    status: job.status,
    createdMs: job.created_ms,
    updatedMs: job.updated_ms,
    startedMs: job.started_ms,
    finishedMs: job.finished_ms,
    partCount: job.total_parts,
    completedParts: job.completed_parts,
    txId: job.last_tx_id,
    txIds: [...(job.tx_ids || [])],
    error: job.error,
    messagePreview: job.message_preview,
  };
}

function isPayloadTooLargeError(error) {
  const text = String(error?.message || error || "").toLowerCase();
  return (
    text.includes("storage mass") ||
    text.includes("larger than max allowed size") ||
    text.includes("transaction is not standard")
  );
}

function isRetryableHandshakeProcessingError(error) {
  const text = String(error?.message || error || "").toLowerCase();
  return text.includes("missing sender address");
}

function truncateMessage(
  content,
  maxLength = DEFAULT_CONTEXTUAL_MESSAGE_TARGET_CHARS,
  { annotateParts = true } = {}
) {
  const text = String(content || "");
  if (text.length <= maxLength) {
    return [text];
  }

  const indicatorReserve = 10;
  const fenceClose = "\n```";
  const chunks = [];
  let remaining = text;
  let carryLang = null;

  while (remaining) {
    const prefix = carryLang != null ? `\`\`\`${carryLang}\n` : "";
    let headroom =
      maxLength - indicatorReserve - prefix.length - fenceClose.length;
    if (headroom < 1) {
      headroom = Math.max(1, Math.floor(maxLength / 2));
    }

    if (prefix.length + remaining.length <= maxLength - indicatorReserve) {
      chunks.push(prefix + remaining);
      break;
    }

    const region = remaining.slice(0, headroom);
    let splitAt = region.lastIndexOf("\n");
    if (splitAt < Math.floor(headroom / 2)) {
      splitAt = region.lastIndexOf(" ");
    }
    if (splitAt < 1) {
      splitAt = headroom;
    }

    const candidate = remaining.slice(0, splitAt);
    const backtickCount =
      (candidate.match(/`/g) || []).length -
      (candidate.match(/\\`/g) || []).length;
    if (backtickCount % 2 === 1) {
      let lastBacktick = candidate.lastIndexOf("`");
      while (lastBacktick > 0 && candidate[lastBacktick - 1] === "\\") {
        lastBacktick = candidate.lastIndexOf("`", lastBacktick - 1);
      }
      if (lastBacktick > 0) {
        const safeSpace = candidate.lastIndexOf(" ", lastBacktick);
        const safeNewline = candidate.lastIndexOf("\n", lastBacktick);
        const safeSplit = Math.max(safeSpace, safeNewline);
        if (safeSplit > Math.floor(headroom / 4)) {
          splitAt = safeSplit;
        }
      }
    }

    const chunkBody = remaining.slice(0, splitAt);
    remaining = remaining.slice(splitAt).trimStart();

    let fullChunk = prefix + chunkBody;
    let inCode = carryLang != null;
    let lang = carryLang || "";
    for (const line of chunkBody.split("\n")) {
      const stripped = line.trim();
      if (stripped.startsWith("```")) {
        if (inCode) {
          inCode = false;
          lang = "";
        } else {
          inCode = true;
          const tag = stripped.slice(3).trim();
          lang = tag ? tag.split(/\s+/, 1)[0] : "";
        }
      }
    }

    if (inCode) {
      fullChunk += fenceClose;
      carryLang = lang;
    } else {
      carryLang = null;
    }

    chunks.push(fullChunk);
  }

  if (annotateParts && chunks.length > 1) {
    const total = chunks.length;
    return chunks.map((chunk, index) => `${chunk} (${index + 1}/${total})`);
  }

  return chunks;
}

export class KasiaBridgeCore {
  constructor({
    stateDir,
    indexerUrl,
    nodeUrl,
    network,
    seedPhrase,
    walletClient,
    fetchImpl,
    logger = console,
    maxQueueSize = 100,
    pollLimit = 50,
    processedTxLimit = 1000,
    contextualMessageTargetChars = DEFAULT_CONTEXTUAL_MESSAGE_TARGET_CHARS,
    maxMultipartParts = DEFAULT_CONTEXTUAL_MESSAGE_MAX_PARTS,
    maxSendJobs = DEFAULT_MAX_SEND_JOBS,
    randomBytesFn,
    nowFn = () => Date.now(),
  }) {
    this.stateDir = stateDir;
    this.statePath = join(stateDir, "state.json");
    this.indexerUrl = indexerUrl;
    this.nodeUrl = nodeUrl;
    this.network = network || "mainnet";
    this.seedPhrase = seedPhrase;
    this.fetchImpl = fetchImpl || fetch;
    this.logger = logger;
    this.maxQueueSize = maxQueueSize;
    this.pollLimit = pollLimit;
    this.processedTxLimit = processedTxLimit;
    this.contextualMessageTargetChars = contextualMessageTargetChars;
    this.maxMultipartParts = maxMultipartParts;
    this.maxSendJobs = maxSendJobs;
    this.randomBytesFn = randomBytesFn;
    this.nowFn = nowFn;

    this.walletClient =
      walletClient ||
      new KaspaWalletClient({
        seedPhrase,
        nodeUrl,
        network: this.network,
      });

    this.state = createEmptyState();
    this.messageQueue = [];
    this.walletInfo = null;
    this._sendJobTail = Promise.resolve();
    this._sendJobWaiters = new Map();
    this._closing = false;
  }

  async init() {
    this.walletInfo = await this.walletClient.init();
    this.state = await loadState(this.statePath, {
      address: this.walletInfo.address,
      public_key: this.walletInfo.publicKeyHex,
      network: this.walletInfo.network,
    });
    this.walletClient.loadSendState?.(this.state.wallet.send_state || {});
    await this.walletClient.hydrateSendState?.();
    this._markInterruptedSendJobs();
    await this._saveState();

    try {
      await this.syncOnce();
    } catch (error) {
      this.logger.warn?.(
        `[kasia-bridge] Initial sync failed: ${error?.message || error}`
      );
    }
  }

  async close() {
    this._closing = true;
    await this._saveState();
    await this.walletClient.close?.();
  }

  health() {
    const sendState = this.walletClient.exportSendState?.() || {};
    const activeSendJobCount = Object.values(this.state.send_jobs || {}).filter(
      (job) => !isTerminalSendJobStatus(job.status)
    ).length;
    return {
      status: this.walletClient.isConnected ? "connected" : "starting",
      walletAddress: this.state.wallet.address,
      network: this.state.wallet.network || this.network,
      indexerUrl: this.indexerUrl,
      nodeUrl: this.nodeUrl,
      lastSyncMs: this.state.last_sync_ms,
      pendingOutputCount: Array.isArray(sendState.pending_outputs)
        ? sendState.pending_outputs.length
        : 0,
      reservedOutpointCount: Array.isArray(sendState.reserved_outpoints)
        ? sendState.reserved_outpoints.length
        : 0,
      activeSendJobCount,
    };
  }

  dequeueMessages() {
    return this.messageQueue.splice(0, this.messageQueue.length);
  }

  getChatInfo(chatId) {
    const conversation = this.state.conversations[String(chatId).trim()];
    return {
      name: shortenAddress(conversation?.peer_address || chatId),
      type: "dm",
      chat_id: String(chatId).trim(),
    };
  }

  getSendJob(jobId) {
    return toPublicSendJob(
      this.state.send_jobs[String(jobId || "").trim()] || null
    );
  }

  async respondToHandshake(chatId) {
    const conversation = this.state.conversations[String(chatId).trim()];
    if (!conversation) {
      throw new Error(`No Kasia conversation found for ${chatId}`);
    }
    if (!conversation.their_alias) {
      throw new Error(`Conversation ${chatId} has no peer alias yet`);
    }
    if (!conversation.our_alias) {
      conversation.our_alias = generateAlias(this.randomBytesFn);
    }

    if (!conversation.pending_handshake && conversation.status === "active") {
      return {
        status: "already_active",
        chatId: conversation.peer_address,
      };
    }

    const handshakePayload = buildHandshakePayload({
      alias: conversation.our_alias,
      theirAlias: conversation.their_alias,
      isResponse: true,
      timestamp: this.nowFn(),
    });
    const payloadBytes = buildHandshakeTransactionPayload({
      recipientAddress: conversation.peer_address,
      payload: handshakePayload,
      randomBytesFn: this.randomBytesFn,
    });
    const txId = await this.walletClient.sendPayloadTransaction({
      destinationAddress: conversation.peer_address,
      amountSompi: MINIMUM_MESSAGE_AMOUNT_SOMPI,
      payloadBytes,
    });

    conversation.status = "active";
    conversation.pending_handshake = null;
    touchConversation(conversation, new Date(this.nowFn()).toISOString());
    await this._saveState();

    return {
      status: "sent",
      txId: txId.txId || txId,
      chatId: conversation.peer_address,
    };
  }

  async send({ chatId, message, waitMs = 0 }) {
    if (!message || !String(message).trim()) {
      throw new Error("Message is required");
    }

    const conversation = this._requireActiveConversation(chatId);
    const text = String(message).trim();
    const plan = this._planConversationMessage(conversation, text);
    const job = await this._createSendJob({
      chatId: conversation.peer_address,
      message: text,
      totalParts: plan.partCount,
    });

    if (plan.status === "rejected") {
      await this._updateSendJob(job.job_id, {
        status: "rejected",
        error: plan.error,
        finished_ms: this.nowFn(),
      });
      return this.getSendJob(job.job_id);
    }

    this._scheduleSendJob(job.job_id, conversation.peer_address, plan.chunks);

    if (Number(waitMs) > 0) {
      return await this.waitForSendJob(job.job_id, Number(waitMs));
    }

    return this.getSendJob(job.job_id);
  }

  async waitForSendJob(jobId, waitMs = 0) {
    const normalizedJobId = String(jobId || "").trim();
    if (!normalizedJobId) {
      throw new Error("Send job id is required");
    }

    const current = this.getSendJob(normalizedJobId);
    if (!current || waitMs <= 0 || isTerminalSendJobStatus(current.status)) {
      return current;
    }

    return await new Promise((resolve) => {
      const cleanup = () => {
        clearTimeout(timeout);
        const waiters = this._sendJobWaiters.get(normalizedJobId);
        if (waiters) {
          waiters.delete(onUpdate);
          if (waiters.size === 0) {
            this._sendJobWaiters.delete(normalizedJobId);
          }
        }
      };

      const onUpdate = () => {
        const latest = this.getSendJob(normalizedJobId);
        if (!latest || isTerminalSendJobStatus(latest.status)) {
          cleanup();
          resolve(latest);
        }
      };

      const timeout = setTimeout(() => {
        cleanup();
        resolve(this.getSendJob(normalizedJobId));
      }, Math.max(1, Number(waitMs)));

      const waiters = this._sendJobWaiters.get(normalizedJobId) || new Set();
      waiters.add(onUpdate);
      this._sendJobWaiters.set(normalizedJobId, waiters);
      onUpdate();
    });
  }

  _requireActiveConversation(chatId) {
    const conversation = this.state.conversations[String(chatId).trim()];
    if (
      !conversation ||
      conversation.status !== "active" ||
      !conversation.our_alias ||
      !conversation.their_alias
    ) {
      throw new Error(
        `No active Kasia conversation found for ${chatId}. Wait for an inbound handshake before sending.`
      );
    }
    return conversation;
  }

  _planConversationMessage(conversation, text) {
    const trimmed = String(text || "").trim();
    if (!trimmed) {
      return {
        status: "rejected",
        partCount: 0,
        error: "Message is required",
      };
    }

    let targetChars = Math.min(
      Math.max(DEFAULT_CONTEXTUAL_MESSAGE_MIN_CHARS, trimmed.length),
      this.contextualMessageTargetChars
    );

    while (targetChars >= DEFAULT_CONTEXTUAL_MESSAGE_MIN_CHARS) {
      const rawChunks =
        trimmed.length > targetChars
          ? truncateMessage(trimmed, targetChars, { annotateParts: false })
          : [trimmed];

      if (rawChunks.length > this.maxMultipartParts) {
        return {
          status: "rejected",
          partCount: rawChunks.length,
          error: `Message is too long for Kasia delivery. Hermes caps Kasia sends at ${this.maxMultipartParts} parts.`,
        };
      }

      const chunks =
        rawChunks.length > 1
          ? rawChunks.map(
              (chunk, index) => `${chunk} (${index + 1}/${rawChunks.length})`
            )
          : rawChunks;

      if (chunks.every((chunk) => this._conversationChunkFits(conversation, chunk))) {
        return {
          status: "ready",
          chunks,
          partCount: chunks.length,
        };
      }

      if (targetChars === DEFAULT_CONTEXTUAL_MESSAGE_MIN_CHARS) {
        break;
      }
      targetChars = Math.max(
        DEFAULT_CONTEXTUAL_MESSAGE_MIN_CHARS,
        Math.min(targetChars - 1, Math.floor(targetChars * 0.75))
      );
    }

    return {
      status: "rejected",
      partCount: 0,
      error:
        "Message is too large for Kasia delivery after preflight sizing. Please shorten it and try again.",
    };
  }

  _conversationChunkFits(conversation, message) {
    try {
      const payloadBytes = buildContextualMessageTransactionPayload({
        recipientAddress: conversation.peer_address,
        alias: conversation.our_alias,
        message: String(message),
        randomBytesFn: this.randomBytesFn,
      });
      if (typeof this.walletClient.canFitContextualPayload === "function") {
        return this.walletClient.canFitContextualPayload(payloadBytes);
      }
      return String(message).length <= this.contextualMessageTargetChars;
    } catch {
      return false;
    }
  }

  async _createSendJob({ chatId, message, totalParts }) {
    const nowMs = this.nowFn();
    const job = {
      job_id: randomUUID(),
      chat_id: String(chatId || "").trim() || null,
      status: "queued",
      created_ms: nowMs,
      updated_ms: nowMs,
      started_ms: null,
      finished_ms: null,
      total_parts: Number(totalParts || 0),
      completed_parts: 0,
      tx_ids: [],
      last_tx_id: null,
      error: null,
      message_preview: buildSendJobPreview(message),
    };
    this.state.send_jobs[job.job_id] = job;
    this._pruneSendJobs();
    await this._saveState();
    this._notifySendJobWaiters(job.job_id);
    return job;
  }

  async _updateSendJob(jobId, updates = {}) {
    const job = this.state.send_jobs[String(jobId || "").trim()];
    if (!job) {
      return null;
    }

    Object.assign(job, updates);
    job.updated_ms = this.nowFn();
    job.total_parts = Math.max(0, Number(job.total_parts || 0));
    job.completed_parts = Math.max(
      0,
      Math.min(job.total_parts || 0, Number(job.completed_parts || 0))
    );
    job.tx_ids = Array.isArray(job.tx_ids)
      ? job.tx_ids.filter((value) => typeof value === "string" && value.trim())
      : [];
    job.last_tx_id =
      String(job.last_tx_id || "").trim() || job.tx_ids[job.tx_ids.length - 1] || null;
    if (isTerminalSendJobStatus(job.status) && job.finished_ms == null) {
      job.finished_ms = this.nowFn();
    }

    this._pruneSendJobs();
    await this._saveState();
    this._notifySendJobWaiters(job.job_id);
    return job;
  }

  _pruneSendJobs() {
    const entries = Object.entries(this.state.send_jobs || {});
    if (entries.length <= this.maxSendJobs) {
      return;
    }

    entries
      .sort(
        (left, right) =>
          Number(left[1]?.created_ms || 0) - Number(right[1]?.created_ms || 0)
      )
      .slice(0, entries.length - this.maxSendJobs)
      .forEach(([jobId]) => {
        delete this.state.send_jobs[jobId];
      });
  }

  _markInterruptedSendJobs() {
    const nowMs = this.nowFn();
    for (const job of Object.values(this.state.send_jobs || {})) {
      if (isTerminalSendJobStatus(job.status)) {
        continue;
      }
      job.status = "failed";
      job.error = "Kasia bridge restarted before the send job completed.";
      job.updated_ms = nowMs;
      job.finished_ms = nowMs;
    }
  }

  _notifySendJobWaiters(jobId) {
    const waiters = this._sendJobWaiters.get(jobId);
    if (!waiters || waiters.size === 0) {
      return;
    }
    for (const waiter of [...waiters]) {
      try {
        waiter();
      } catch {}
    }
  }

  _scheduleSendJob(jobId, chatId, chunks) {
    const previous = this._sendJobTail;
    const task = (async () => {
      await previous.catch(() => {});
      if (this._closing) {
        await this._updateSendJob(jobId, {
          status: "failed",
          error: "Kasia bridge stopped before the send job completed.",
        });
        return;
      }

      try {
        await this._runSendJob(jobId, chatId, chunks);
      } catch (error) {
        await this._updateSendJob(jobId, {
          status: "failed",
          error: error?.message || String(error),
        });
      }
    })();

    this._sendJobTail = task.then(() => {}, () => {});
    return task;
  }

  async _runSendJob(jobId, chatId, chunks) {
    const conversation = this._requireActiveConversation(chatId);
    await this._updateSendJob(jobId, {
      status: "running",
      started_ms: this.nowFn(),
      total_parts: chunks.length,
      completed_parts: 0,
      error: null,
    });

    const txResults = [];
    for (let index = 0; index < chunks.length; index += 1) {
      const result = await this._sendConversationMessageChunk(
        conversation,
        chunks[index]
      );
      const txId = result?.txId || result?.messageId || result;
      txResults.push(result);
      await this._updateSendJob(jobId, {
        status: "running",
        completed_parts: index + 1,
        tx_ids: txResults
          .map((entry) => entry?.txId || entry?.messageId || entry)
          .filter(Boolean),
        last_tx_id: txId || null,
      });
    }

    touchConversation(conversation, new Date(this.nowFn()).toISOString());
    await this._updateSendJob(jobId, {
      status: "sent",
      completed_parts: chunks.length,
      tx_ids: txResults
        .map((entry) => entry?.txId || entry?.messageId || entry)
        .filter(Boolean),
    });
  }

  async _sendConversationMessageChunk(conversation, message) {
    const payloadBytes = buildContextualMessageTransactionPayload({
      recipientAddress: conversation.peer_address,
      alias: conversation.our_alias,
      message: String(message),
      randomBytesFn: this.randomBytesFn,
    });
    return await this.walletClient.sendPayloadTransaction({
      destinationAddress: this.walletInfo.address,
      amountSompi: MINIMUM_MESSAGE_AMOUNT_SOMPI,
      payloadBytes,
      strategy: "contextual",
    });
  }

  async syncOnce() {
    await this._pollHandshakes();
    await this._pollContextualMessages();
    this.state.last_sync_ms = this.nowFn();
    await this._saveState();
  }

  async _pollHandshakes() {
    const records = await this._fetchJson("/handshakes/by-receiver", {
      address: this.walletInfo.address,
      block_time: this.state.cursors.handshakes_block_time,
      limit: this.pollLimit,
    });

    let maxBlockTime = Number(this.state.cursors.handshakes_block_time || 0);
    for (const record of records) {
      maxBlockTime = Math.max(maxBlockTime, Number(record.block_time || 0));
      if (!record?.tx_id || hasProcessedTx(this.state, record.tx_id)) {
        continue;
      }
      let shouldMarkProcessed = true;
      try {
        const plaintext = decryptSealedMessage(
          this.walletInfo.privateKeyHex,
          record.message_payload
        );
        const payload = parseHandshakePayload(plaintext);
        this._handleHandshakeRecord(record, payload);
      } catch (error) {
        this.logger.warn?.(
          `[kasia-bridge] Failed to process handshake ${record?.tx_id}: ${error?.message || error}`
        );
        shouldMarkProcessed = !isRetryableHandshakeProcessingError(error);
      }
      if (shouldMarkProcessed) {
        markProcessedTx(this.state, record.tx_id, this.processedTxLimit);
      }
    }

    this.state.cursors.handshakes_block_time = maxBlockTime;
  }

  _handleHandshakeRecord(record, payload) {
    const peerAddress = String(record.sender || "").trim();
    if (!peerAddress) {
      throw new Error("Handshake record is missing sender address");
    }

    const conversation = ensureConversation(this.state, peerAddress);
    conversation.last_handshake_block_time = Math.max(
      Number(conversation.last_handshake_block_time || 0),
      Number(record.block_time || 0)
    );
    touchConversation(conversation, new Date(this.nowFn()).toISOString());

    if (payload.isResponse) {
      if (payload.theirAlias) {
        conversation.our_alias = payload.theirAlias;
      }
      conversation.their_alias = payload.alias;
      if (conversation.our_alias && conversation.their_alias) {
        conversation.status = "active";
      }
      conversation.pending_handshake = null;
      return;
    }

    conversation.their_alias = payload.alias;
    conversation.our_alias =
      conversation.our_alias || generateAlias(this.randomBytesFn);
    conversation.status = "pending";
    conversation.pending_handshake = {
      tx_id: record.tx_id,
      block_time: Number(record.block_time || 0),
      received_at: new Date(this.nowFn()).toISOString(),
    };

    this._enqueueEvent({
      eventType: "handshake_request",
      messageId: record.tx_id,
      chatId: peerAddress,
      senderId: peerAddress,
      senderName: shortenAddress(peerAddress),
      body: "Handshake request",
      timestampMs: Number(record.block_time || this.nowFn()),
      raw: {
        ...record,
        payload,
      },
    });
  }

  async _pollContextualMessages() {
    for (const conversation of Object.values(this.state.conversations)) {
      if (
        conversation.status !== "active" ||
        !conversation.their_alias ||
        !conversation.peer_address
      ) {
        continue;
      }

      const records = await this._fetchJson("/contextual-messages/by-sender", {
        address: conversation.peer_address,
        alias: encodeIndexerAlias(conversation.their_alias),
        block_time: conversation.last_context_block_time || 0,
        limit: this.pollLimit,
      });

      let maxBlockTime = Number(conversation.last_context_block_time || 0);
      for (const record of records) {
        maxBlockTime = Math.max(maxBlockTime, Number(record.block_time || 0));
        if (!record?.tx_id || hasProcessedTx(this.state, record.tx_id)) {
          continue;
        }
        try {
          const sealedMessage = decodeIndexedContextualMessagePayload(
            record.message_payload
          );
          const messageBody = decryptSealedMessage(
            this.walletInfo.privateKeyHex,
            sealedMessage
          );
          touchConversation(conversation, new Date(this.nowFn()).toISOString());
          this._enqueueEvent({
            eventType: "message",
            messageId: record.tx_id,
            chatId: conversation.peer_address,
            senderId: conversation.peer_address,
            senderName: shortenAddress(conversation.peer_address),
            body: messageBody,
            timestampMs: Number(record.block_time || this.nowFn()),
            raw: record,
          });
        } catch (error) {
          this.logger.warn?.(
            `[kasia-bridge] Failed to process message ${record?.tx_id}: ${error?.message || error}`
          );
        }
        markProcessedTx(this.state, record.tx_id, this.processedTxLimit);
      }

      conversation.last_context_block_time = maxBlockTime;
    }
  }

  async _fetchJson(path, params) {
    const url = joinUrl(this.indexerUrl, path, params);
    const response = await this.fetchImpl(url, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(
        `Indexer request failed (${response.status}) for ${url.pathname}: ${body}`
      );
    }

    return await response.json();
  }

  _enqueueEvent(event) {
    this.messageQueue.push(event);
    if (this.messageQueue.length > this.maxQueueSize) {
      this.messageQueue.splice(0, this.messageQueue.length - this.maxQueueSize);
    }
  }

  async _saveState() {
    this.state.wallet.send_state = this.walletClient.exportSendState?.() || {};
    await saveState(this.statePath, this.state);
  }
}
