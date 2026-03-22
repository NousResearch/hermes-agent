import { randomUUID } from "node:crypto";
import { join } from "node:path";

import {
  buildBroadcastTransactionPayload,
  buildContextualMessageTransactionPayload,
  buildHandshakePayload,
  buildHandshakeTransactionPayload,
  decodeIndexedContextualMessagePayload,
  decryptSealedMessage,
  generateAlias,
  MINIMUM_MESSAGE_AMOUNT_SOMPI,
  normalizeBroadcastChannelName,
  parseBroadcastPayload,
  parseContextualMessagePayload,
  parseHandshakePayload,
  shortenAddress,
} from "./protocol.js";
import { EndpointPool } from "./endpoint_pool.js";
import { KnsClient, normalizeKnsName } from "./kns_client.js";
import {
  createEmptyState,
  ensureBroadcastChannel,
  ensureConversation,
  hasProcessedTx,
  loadState,
  markProcessedTx,
  saveState,
  touchBroadcastChannel,
  touchConversation,
} from "./state.js";
import {
  buildSendJobPreview,
  DEFAULT_CONTEXTUAL_MESSAGE_MAX_PARTS,
  DEFAULT_CONTEXTUAL_MESSAGE_MIN_CHARS,
  DEFAULT_CONTEXTUAL_MESSAGE_TARGET_CHARS,
  DEFAULT_IDENTITY_REFRESH_MS,
  DEFAULT_LIVE_LOOKBACK_MS,
  DEFAULT_MAX_SEND_JOBS,
  DEFAULT_NODE_STARTUP_TIMEOUT_MS,
  DEFAULT_SEND_JOB_INDEXER_LOOKBACK_MS,
  encodeIndexerAlias,
  isBlockingSendJobStatus,
  isIndexerTrackedSendJobStatus,
  isPayloadTooLargeError,
  isRetryableHandshakeProcessingError,
  isRetryableNodeError,
  isTerminalSendJobStatus,
  joinUrl,
  mempoolEntriesForAddress,
  parseEndpointList,
  publicDelivery,
  senderTransactionsFromAddressEntry,
  toPublicSendJob,
  truncateMessage,
  txBlockTimeFromRecord,
  txIdFromTransaction,
  txPayloadFromTransaction,
  withTimeout,
} from "./bridge_core_support.js";
import { KaspaWalletClient } from "./kaspa_wallet.js";

const RECOMMENDED_MIN_WALLET_BALANCE_SOMPI = MINIMUM_MESSAGE_AMOUNT_SOMPI * 2n;

function toBigIntOrZero(value) {
  if (typeof value === "bigint") {
    return value;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return BigInt(Math.trunc(value));
  }
  const normalized = String(value ?? "").trim();
  if (!normalized) {
    return 0n;
  }
  try {
    return BigInt(normalized);
  } catch {
    return 0n;
  }
}

export class KasiaBridgeCore {
  constructor({
    stateDir,
    indexerUrl,
    indexerUrls,
    nodeUrl,
    nodeUrls,
    network,
    seedPhrase,
    knsUrl,
    feePolicy = "auto",
    broadcastSubscriptions = {},
    allowedBroadcastChannels = [],
    allowAllBroadcastChannels = false,
    walletClient,
    knsClient,
    fetchImpl,
    logger = console,
    maxQueueSize = 100,
    pollLimit = 50,
    processedTxLimit = 1000,
    contextualMessageTargetChars = DEFAULT_CONTEXTUAL_MESSAGE_TARGET_CHARS,
    respectContextualMessageTarget =
      contextualMessageTargetChars !== DEFAULT_CONTEXTUAL_MESSAGE_TARGET_CHARS,
    maxMultipartParts = DEFAULT_CONTEXTUAL_MESSAGE_MAX_PARTS,
    maxSendJobs = DEFAULT_MAX_SEND_JOBS,
    randomBytesFn,
    nowFn = () => Date.now(),
  }) {
    this.stateDir = stateDir;
    this.statePath = join(stateDir, "state.json");
    this.network = network || "mainnet";
    this.seedPhrase = seedPhrase;
    this.knsUrl = String(knsUrl || "").trim();
    this.feePolicy = String(feePolicy || "auto");
    this.fetchImpl = fetchImpl || fetch;
    this.logger = logger;
    this.maxQueueSize = maxQueueSize;
    this.pollLimit = pollLimit;
    this.processedTxLimit = processedTxLimit;
    this.contextualMessageTargetChars = contextualMessageTargetChars;
    this.respectContextualMessageTarget = Boolean(respectContextualMessageTarget);
    this.maxMultipartParts = maxMultipartParts;
    this.maxSendJobs = maxSendJobs;
    this.randomBytesFn = randomBytesFn;
    this.nowFn = nowFn;
    this.identityRefreshMs = DEFAULT_IDENTITY_REFRESH_MS;
    this.allowAllBroadcastChannels = Boolean(allowAllBroadcastChannels);
    this.allowedBroadcastChannels = new Set(
      parseEndpointList(allowedBroadcastChannels).map((value) =>
        normalizeBroadcastChannelName(value)
      )
    );
    this.broadcastSubscriptions = Object.entries(broadcastSubscriptions || {}).reduce(
      (acc, [channelName, publishers]) => {
        const normalizedName = normalizeBroadcastChannelName(channelName);
        acc[normalizedName] = parseEndpointList(publishers);
        return acc;
      },
      {}
    );
    this.indexerPool = new EndpointPool({
      urls: parseEndpointList(indexerUrls?.length ? indexerUrls : [indexerUrl]),
      name: "indexer",
      nowFn,
    });
    this.nodePool = new EndpointPool({
      urls: parseEndpointList(nodeUrls?.length ? nodeUrls : [nodeUrl]),
      name: "node",
      nowFn,
    });

    this.walletClient =
      walletClient ||
      new KaspaWalletClient({
        seedPhrase,
        nodeUrl: this.nodePool.activeUrl,
        network: this.network,
        feePolicy: this.feePolicy,
      });
    this.knsClient =
      knsClient ||
      new KnsClient({
        baseUrl: this.knsUrl,
        network: this.network,
        fetchImpl: this.fetchImpl,
        nowFn: this.nowFn,
      });

    this.state = createEmptyState();
    this.messageQueue = [];
    this.walletInfo = null;
    this._sendJobTail = Promise.resolve();
    this._sendJobWaiters = new Map();
    this._saveStateTail = Promise.resolve();
    this._closing = false;
  }

  async init({ skipInitialSync = false } = {}) {
    this.walletInfo = await this._initWalletClient();
    this.state = await loadState(this.statePath, {
      address: this.walletInfo.address,
      public_key: this.walletInfo.publicKeyHex,
      network: this.walletInfo.network,
    });
    this._hydrateConfiguredBroadcastChannels();
    this.walletClient.loadSendState?.(this.state.wallet.send_state || {});
    await this.walletClient.hydrateSendState?.();
    this._markInterruptedSendJobs();
    await this._refreshConversationIdentities({ force: false });
    await this._saveState();

    if (skipInitialSync) {
      return;
    }

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
    const jobs = Object.values(this.state.send_jobs || {});
    const activeSendJobCount = jobs.filter(
      (job) => !isTerminalSendJobStatus(job.status)
    ).length;
    const waitingForIndexerCount = jobs.filter(
      (job) => job.status === "submitted" || job.status === "waiting_for_indexer"
    ).length;
    const indexerSnapshot = this.indexerPool.snapshot();
    const nodeSnapshot = this.nodePool.snapshot();
    const balanceSnapshot = this.walletClient.getBalanceSnapshot?.() || {};
    const onChainBalanceSompi = toBigIntOrZero(balanceSnapshot.onChainBalanceSompi);
    const availableMatureBalanceSompi = toBigIntOrZero(
      balanceSnapshot.availableMatureBalanceSompi
    );
    const availablePendingBalanceSompi = toBigIntOrZero(
      balanceSnapshot.availablePendingBalanceSompi
    );
    let walletFundingState = "ready";
    if (onChainBalanceSompi <= 0n) {
      walletFundingState = "unfunded";
    } else if (
      availableMatureBalanceSompi < MINIMUM_MESSAGE_AMOUNT_SOMPI ||
      onChainBalanceSompi < RECOMMENDED_MIN_WALLET_BALANCE_SOMPI
    ) {
      walletFundingState = "low";
    }
    return {
      status: this.walletClient.isConnected ? "connected" : "starting",
      walletAddress: this.state.wallet.address,
      walletBalanceSompi: String(onChainBalanceSompi),
      availableMatureBalanceSompi: String(availableMatureBalanceSompi),
      availablePendingBalanceSompi: String(availablePendingBalanceSompi),
      recommendedMinBalanceSompi: String(RECOMMENDED_MIN_WALLET_BALANCE_SOMPI),
      minimumMessageAmountSompi: String(MINIMUM_MESSAGE_AMOUNT_SOMPI),
      walletFundingState,
      matureUtxoCount: Number(balanceSnapshot.matureUtxoCount || 0),
      pendingUtxoCount: Number(balanceSnapshot.pendingUtxoCount || 0),
      trackedPendingUtxoCount: Number(balanceSnapshot.trackedPendingUtxoCount || 0),
      walletBalanceUpdatedAtMs: Number(balanceSnapshot.updatedAtMs || 0),
      network: this.state.wallet.network || this.network,
      indexerUrl: indexerSnapshot.activeUrl,
      nodeUrl: nodeSnapshot.activeUrl || this.walletClient.getNodeUrl?.() || null,
      indexerPool: indexerSnapshot,
      nodePool: nodeSnapshot,
      knsUrl: this.knsClient.isEnabled() ? this.knsClient.baseUrl : null,
      feePolicy: this.walletClient.getFeePolicy?.() || this.feePolicy,
      feeRateSompiPerGram: this.walletClient.getLastResolvedFeeRate?.() || null,
      lastSyncMs: this.state.last_sync_ms,
      pendingOutputCount: Array.isArray(sendState.pending_outputs)
        ? sendState.pending_outputs.length
        : 0,
      reservedOutpointCount: Array.isArray(sendState.reserved_outpoints)
        ? sendState.reserved_outpoints.length
        : 0,
      activeSendJobCount,
      waitingForIndexerCount,
      broadcastChannelCount: Object.keys(this.state.broadcasts?.channels || {}).length,
    };
  }

  dequeueMessages() {
    return this.messageQueue.splice(0, this.messageQueue.length);
  }

  getChatInfo(chatId) {
    const normalizedChatId = String(chatId).trim();
    if (normalizedChatId.startsWith("broadcast:")) {
      const channelName = normalizedChatId.slice("broadcast:".length).trim().toLowerCase();
      const channel = this.state.broadcasts?.channels?.[channelName];
      return {
        name: `#${channelName}`,
        type: "channel",
        chat_id: channel?.channel_id || `broadcast:${channelName}`,
      };
    }

    const conversation = this.state.conversations[normalizedChatId];
    const displayName =
      conversation?.display_name ||
      conversation?.nickname ||
      conversation?.kns_name ||
      shortenAddress(conversation?.peer_address || normalizedChatId);
    return {
      name: displayName,
      type: "dm",
      chat_id: normalizedChatId,
    };
  }

  getSendJob(jobId) {
    return toPublicSendJob(
      this.state.send_jobs[String(jobId || "").trim()] || null
    );
  }

  async resolveTarget(target) {
    const rawTarget = String(target || "").trim();
    if (!rawTarget) {
      throw new Error("Kasia target is required");
    }
    if (rawTarget.startsWith("broadcast:")) {
      const channelName = normalizeBroadcastChannelName(
        rawTarget.slice("broadcast:".length)
      );
      const channel = ensureBroadcastChannel(this.state, channelName, {
        publishers: this.broadcastSubscriptions[channelName] || [],
        allow_publish:
          this.allowAllBroadcastChannels ||
          this.allowedBroadcastChannels.has(channelName),
      });
      return {
        kind: "broadcast",
        chatId: channel.channel_id,
        channelName,
      };
    }
    if (rawTarget.startsWith("#")) {
      const channelName = normalizeBroadcastChannelName(rawTarget.slice(1));
      const channel = ensureBroadcastChannel(this.state, channelName, {
        publishers: this.broadcastSubscriptions[channelName] || [],
        allow_publish:
          this.allowAllBroadcastChannels ||
          this.allowedBroadcastChannels.has(channelName),
      });
      return {
        kind: "broadcast",
        chatId: channel.channel_id,
        channelName,
      };
    }

    const resolvedKnsAddress = await this.knsClient.resolveTarget(
      rawTarget,
      this.state.kns_cache
    );
    const peerAddress = resolvedKnsAddress || rawTarget;
    return {
      kind: "dm",
      chatId: peerAddress,
      peerAddress,
      knsName: resolvedKnsAddress ? normalizeKnsName(rawTarget) : null,
    };
  }

  async initiateHandshake({ chatId, retry = false, displayName = null }) {
    const resolved = await this.resolveTarget(chatId);
    if (resolved.kind !== "dm") {
      throw new Error("Handshake initiation is only supported for direct Kasia conversations");
    }

    const conversation = ensureConversation(this.state, resolved.peerAddress);
    if (displayName) {
      conversation.nickname = String(displayName).trim() || conversation.nickname;
    }
    if (resolved.knsName) {
      conversation.kns_name = resolved.knsName;
      conversation.display_name = conversation.nickname || resolved.knsName;
      conversation.identity_source = conversation.nickname ? "nickname" : "kns";
    }
    if (!conversation.our_alias) {
      conversation.our_alias = generateAlias(this.randomBytesFn);
    }
    if (
      conversation.status === "active" &&
      conversation.our_alias &&
      conversation.their_alias
    ) {
      await this._saveState();
      return {
        status: "already_active",
        chatId: conversation.peer_address,
      };
    }
    if (conversation.pending_outbound_handshake && !retry) {
      return {
        status: "pending",
        chatId: conversation.peer_address,
        txId: conversation.pending_outbound_handshake.tx_id || null,
      };
    }

    const handshakePayload = buildHandshakePayload({
      alias: conversation.our_alias,
      timestamp: this.nowFn(),
    });
    const payloadBytes = buildHandshakeTransactionPayload({
      recipientAddress: conversation.peer_address,
      payload: handshakePayload,
      randomBytesFn: this.randomBytesFn,
    });
    const txId = await this._withWalletOperation(() =>
      this.walletClient.sendPayloadTransaction({
        destinationAddress: conversation.peer_address,
        amountSompi: MINIMUM_MESSAGE_AMOUNT_SOMPI,
        payloadBytes,
        strategy: "direct",
      })
    );

    conversation.status = "initiated";
    conversation.pending_outbound_handshake = {
      tx_id: txId.txId || txId,
      sent_at: new Date(this.nowFn()).toISOString(),
      retry_count: retry
        ? Number(conversation.pending_outbound_handshake?.retry_count || 0) + 1
        : Number(conversation.pending_outbound_handshake?.retry_count || 0),
    };
    touchConversation(conversation, new Date(this.nowFn()).toISOString());
    await this._refreshConversationIdentity(conversation, { force: false });
    await this._saveState();

    return {
      status: "sent",
      txId: txId.txId || txId,
      chatId: conversation.peer_address,
    };
  }

  async subscribeBroadcastChannel({ channelName, publishers = [] }) {
    const normalizedName = normalizeBroadcastChannelName(channelName);
    const channel = ensureBroadcastChannel(this.state, normalizedName, {
      publishers:
        parseEndpointList(publishers).length > 0
          ? parseEndpointList(publishers)
          : this.broadcastSubscriptions[normalizedName] || [],
      allow_publish:
        this.allowAllBroadcastChannels ||
        this.allowedBroadcastChannels.has(normalizedName),
    });
    channel.publishers = [
      ...new Set([
        ...channel.publishers,
        ...parseEndpointList(publishers),
      ]),
    ];
    touchBroadcastChannel(channel, new Date(this.nowFn()).toISOString());
    await this._saveState();
    return {
      status: "subscribed",
      chatId: channel.channel_id,
      channelName: channel.channel_name,
      publishers: [...channel.publishers],
    };
  }

  async respondToHandshake(chatId) {
    const resolved = await this.resolveTarget(chatId);
    if (resolved.kind !== "dm") {
      throw new Error("Handshake responses are only supported for direct Kasia conversations");
    }
    const conversation = this.state.conversations[String(resolved.chatId).trim()];
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
    const txId = await this._withWalletOperation(() =>
      this.walletClient.sendPayloadTransaction({
        destinationAddress: conversation.peer_address,
        amountSompi: MINIMUM_MESSAGE_AMOUNT_SOMPI,
        payloadBytes,
      })
    );

    conversation.status = "active";
    conversation.pending_handshake = null;
    conversation.pending_outbound_handshake = null;
    touchConversation(conversation, new Date(this.nowFn()).toISOString());
    await this._refreshConversationIdentity(conversation, { force: false });
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

    const resolved = await this.resolveTarget(chatId);
    if (resolved.kind === "broadcast") {
      return await this.sendBroadcast({
        channelName: resolved.channelName,
        message,
        waitMs,
      });
    }

    const conversation = this._requireActiveConversation(resolved.chatId);
    const text = String(message).trim();
    const plan = this._planConversationMessage(conversation, text);
    const job = await this._createSendJob({
      chatId: conversation.peer_address,
      message: text,
      totalParts: plan.partCount,
      jobKind: "dm",
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
    if (
      !current ||
      waitMs <= 0 ||
      isTerminalSendJobStatus(current.status) ||
      !isBlockingSendJobStatus(current.status)
    ) {
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
        if (
          !latest ||
          isTerminalSendJobStatus(latest.status) ||
          !isBlockingSendJobStatus(latest.status)
        ) {
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
        `No active Kasia conversation found for ${chatId}. Initiate a handshake first or wait for an inbound handshake before sending.`
      );
    }
    return conversation;
  }

  async sendBroadcast({ channelName, message, waitMs = 0 }) {
    const normalizedChannel = normalizeBroadcastChannelName(channelName);
    const channel = ensureBroadcastChannel(this.state, normalizedChannel, {
      publishers: this.broadcastSubscriptions[normalizedChannel] || [],
      allow_publish:
        this.allowAllBroadcastChannels ||
        this.allowedBroadcastChannels.has(normalizedChannel),
    });
    if (!channel.allow_publish) {
      throw new Error(
        `Broadcast publishing is not allowed for #${normalizedChannel}. Add it to KASIA_ALLOWED_BROADCAST_CHANNELS or enable KASIA_ALLOW_ALL_BROADCAST_CHANNELS.`
      );
    }

    const trimmedMessage = String(message || "").trim();
    const payloadBytes = buildBroadcastTransactionPayload({
      channelName: normalizedChannel,
      message: trimmedMessage,
    });
    if (!this.walletClient.canFitContextualPayload(payloadBytes)) {
      throw new Error(
        "Broadcasts must fit in one Kasia on-chain message. Please shorten the message and try again."
      );
    }

    const job = await this._createSendJob({
      chatId: channel.channel_id,
      message: trimmedMessage,
      totalParts: 1,
      jobKind: "broadcast",
    });
    this._scheduleSendJob(job.job_id, channel.channel_id, [trimmedMessage]);

    if (Number(waitMs) > 0) {
      return await this.waitForSendJob(job.job_id, Number(waitMs));
    }
    return this.getSendJob(job.job_id);
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

    const singlePartAllowed =
      !this.respectContextualMessageTarget ||
      trimmed.length <= this.contextualMessageTargetChars;

    if (singlePartAllowed && this._conversationChunkFits(conversation, trimmed)) {
      return {
        status: "ready",
        chunks: [trimmed],
        partCount: 1,
      };
    }

    let targetChars = Math.min(
      trimmed.length,
      this.contextualMessageTargetChars
    );
    if (targetChars >= trimmed.length) {
      targetChars = trimmed.length - 1;
    }
    targetChars = Math.max(DEFAULT_CONTEXTUAL_MESSAGE_MIN_CHARS, targetChars);

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
      targetChars -= 1;
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

  async _createSendJob({ chatId, message, totalParts, jobKind = "dm" }) {
    const nowMs = this.nowFn();
    const job = {
      job_id: randomUUID(),
      chat_id: String(chatId || "").trim() || null,
      status: "queued",
      created_ms: nowMs,
      updated_ms: nowMs,
      started_ms: null,
      finished_ms: null,
      submitted_ms: null,
      indexed_ms: null,
      indexed_block_time_ms: null,
      total_parts: Number(totalParts || 0),
      completed_parts: 0,
      indexed_parts: 0,
      tx_ids: [],
      indexed_tx_ids: [],
      last_tx_id: null,
      error: null,
      message_preview: buildSendJobPreview(message),
      job_kind: jobKind,
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
    job.indexed_parts = Math.max(
      0,
      Math.min(job.total_parts || 0, Number(job.indexed_parts || 0))
    );
    job.tx_ids = Array.isArray(job.tx_ids)
      ? job.tx_ids.filter((value) => typeof value === "string" && value.trim())
      : [];
    job.indexed_tx_ids = Array.isArray(job.indexed_tx_ids)
      ? job.indexed_tx_ids.filter(
          (value) => typeof value === "string" && value.trim()
        )
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
      if (isIndexerTrackedSendJobStatus(job.status) && job.tx_ids.length > 0) {
        job.status = "waiting_for_indexer";
        job.updated_ms = nowMs;
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
    const isBroadcastJob = String(chatId || "").startsWith("broadcast:");
    const conversation = isBroadcastJob ? null : this._requireActiveConversation(chatId);
    await this._updateSendJob(jobId, {
      status: "submitting",
      started_ms: this.nowFn(),
      total_parts: chunks.length,
      completed_parts: 0,
      indexed_parts: 0,
      indexed_tx_ids: [],
      error: null,
    });

    const txResults = [];
    for (let index = 0; index < chunks.length; index += 1) {
      const result = isBroadcastJob
        ? await this._sendBroadcastMessageChunk(chatId, chunks[index])
        : await this._sendConversationMessageChunk(conversation, chunks[index]);
      const txId = result?.txId || result?.messageId || result;
      txResults.push(result);
      await this._updateSendJob(jobId, {
        status: "submitting",
        completed_parts: index + 1,
        tx_ids: txResults
          .map((entry) => entry?.txId || entry?.messageId || entry)
          .filter(Boolean),
        last_tx_id: txId || null,
      });
    }

    if (conversation) {
      touchConversation(conversation, new Date(this.nowFn()).toISOString());
    }
    await this._updateSendJob(jobId, {
      status: "submitted",
      submitted_ms: this.nowFn(),
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
    return await this._withWalletOperation(() =>
      this.walletClient.sendPayloadTransaction({
        destinationAddress: this.walletInfo.address,
        amountSompi: MINIMUM_MESSAGE_AMOUNT_SOMPI,
        payloadBytes,
        strategy: "contextual",
      })
    );
  }

  async _sendBroadcastMessageChunk(chatId, message) {
    const channelName = String(chatId || "").slice("broadcast:".length);
    const payloadBytes = buildBroadcastTransactionPayload({
      channelName,
      message: String(message),
    });
    return await this._withWalletOperation(() =>
      this.walletClient.sendPayloadTransaction({
        destinationAddress: this.walletInfo.address,
        amountSompi: MINIMUM_MESSAGE_AMOUNT_SOMPI,
        payloadBytes,
        strategy: "contextual",
      })
    );
  }

  _hydrateConfiguredBroadcastChannels() {
    for (const [channelName, publishers] of Object.entries(
      this.broadcastSubscriptions || {}
    )) {
      ensureBroadcastChannel(this.state, channelName, {
        publishers,
        allow_publish:
          this.allowAllBroadcastChannels ||
          this.allowedBroadcastChannels.has(channelName),
      });
    }
  }

  async _initWalletClient() {
    const candidates = this.nodePool.getCandidates();
    let lastError = null;
    for (const nodeUrl of candidates) {
      try {
        let walletInfo;
        if (
          this.walletClient.switchNodeUrl &&
          this.walletClient.getNodeUrl?.() &&
          this.walletClient.getNodeUrl() !== nodeUrl
        ) {
          walletInfo = await withTimeout(
            this.walletClient.switchNodeUrl(nodeUrl),
            DEFAULT_NODE_STARTUP_TIMEOUT_MS,
            `Kasia wallet startup via ${nodeUrl}`
          );
        } else {
          walletInfo = await withTimeout(
            this.walletClient.init(),
            DEFAULT_NODE_STARTUP_TIMEOUT_MS,
            `Kasia wallet startup via ${nodeUrl}`
          );
        }
        this.nodePool.markSuccess(nodeUrl);
        return walletInfo || this.walletClient.getWalletInfo?.();
      } catch (error) {
        lastError = error;
        this.nodePool.markFailure(nodeUrl, error?.message || error);
      }
    }
    throw lastError || new Error("Failed to initialize the Kasia wallet client");
  }

  async _withWalletOperation(operation) {
    const candidates = this.nodePool.getCandidates();
    let lastError = null;
    for (const nodeUrl of candidates) {
      try {
        if (
          this.walletClient.switchNodeUrl &&
          this.walletClient.getNodeUrl?.() &&
          this.walletClient.getNodeUrl() !== nodeUrl
        ) {
          await this.walletClient.switchNodeUrl(nodeUrl);
          this.walletInfo = this.walletClient.getWalletInfo();
        }
        const result = await operation(nodeUrl);
        this.nodePool.markSuccess(nodeUrl);
        return result;
      } catch (error) {
        lastError = error;
        if (!isRetryableNodeError(error) || candidates.length === 1) {
          throw error;
        }
        this.nodePool.markFailure(nodeUrl, error?.message || error);
      }
    }
    throw lastError || new Error("Kasia wallet operation failed");
  }

  async _refreshConversationIdentity(conversation, { force = false } = {}) {
    if (!conversation?.peer_address) {
      return;
    }
    if (!this.knsClient.isEnabled()) {
      if (!conversation.display_name) {
        conversation.display_name =
          conversation.nickname || shortenAddress(conversation.peer_address);
        conversation.identity_source = conversation.nickname ? "nickname" : "address";
      }
      return;
    }

    const nowMs = this.nowFn();
    if (
      !force &&
      Number(conversation.last_identity_refresh_ms || 0) + this.identityRefreshMs >
        nowMs
    ) {
      if (!conversation.display_name) {
        conversation.display_name =
          conversation.nickname ||
          conversation.kns_name ||
          shortenAddress(conversation.peer_address);
      }
      return;
    }

    const knsName = await this.knsClient.lookupPrimaryName(
      conversation.peer_address,
      this.state.kns_cache
    ).catch(() => null);
    if (knsName) {
      conversation.kns_name = knsName;
    }
    conversation.display_name =
      conversation.nickname ||
      conversation.kns_name ||
      shortenAddress(conversation.peer_address);
    conversation.identity_source = conversation.nickname
      ? "nickname"
      : conversation.kns_name
      ? "kns"
      : "address";
    conversation.last_identity_refresh_ms = nowMs;
  }

  async _refreshConversationIdentities({ force = false } = {}) {
    for (const conversation of Object.values(this.state.conversations || {})) {
      await this._refreshConversationIdentity(conversation, { force });
    }
  }

  async _pollLiveContextualMessages() {
    if (typeof this.walletClient.getAddressMempoolEntries !== "function") {
      return;
    }
    const conversations = Object.values(this.state.conversations || {}).filter(
      (conversation) =>
        conversation.status === "active" &&
        conversation.their_alias &&
        conversation.peer_address
    );
    if (conversations.length === 0) {
      return;
    }

    let mempool;
    try {
      mempool = await this._withWalletOperation(() =>
        this.walletClient.getAddressMempoolEntries(
          conversations.map((conversation) => conversation.peer_address)
        )
      );
    } catch (error) {
      this.logger.debug?.(
        `[kasia-bridge] Live contextual poll skipped: ${error?.message || error}`
      );
      return;
    }

    for (const conversation of conversations) {
      const addressEntry = mempoolEntriesForAddress(
        mempool,
        conversation.peer_address
      );
      for (const transaction of senderTransactionsFromAddressEntry(addressEntry)) {
        const txId = txIdFromTransaction(transaction);
        if (!txId || hasProcessedTx(this.state, txId)) {
          continue;
        }

        try {
          const parsed = parseContextualMessagePayload(txPayloadFromTransaction(transaction));
          if (parsed.alias !== conversation.their_alias) {
            continue;
          }
          const messageBody = decryptSealedMessage(
            this.walletInfo.privateKeyHex,
            parsed.sealedMessage
          );
          const liveObservedMs = this.nowFn();
          conversation.last_live_tx_seen_ms = Math.max(
            Number(conversation.last_live_tx_seen_ms || 0),
            liveObservedMs
          );
          touchConversation(conversation, new Date(liveObservedMs).toISOString());
          await this._refreshConversationIdentity(conversation, { force: false });
          this._enqueueEvent({
            eventType: "message",
            messageId: txId,
            chatId: conversation.peer_address,
            senderId: conversation.peer_address,
            senderName: conversation.display_name || shortenAddress(conversation.peer_address),
            body: messageBody,
            timestampMs: liveObservedMs,
            raw: {
              transactionId: txId,
              delivery: publicDelivery(null, {
                liveObservedMs,
              }),
            },
          });
          markProcessedTx(this.state, txId, this.processedTxLimit);
        } catch (error) {
          this.logger.debug?.(
            `[kasia-bridge] Live contextual tx ${txId} ignored: ${error?.message || error}`
          );
        }
      }
    }
  }

  async _pollLiveBroadcastMessages() {
    if (typeof this.walletClient.getAddressMempoolEntries !== "function") {
      return;
    }
    const channels = Object.values(this.state.broadcasts?.channels || {}).filter(
      (channel) => Array.isArray(channel.publishers) && channel.publishers.length > 0
    );
    if (channels.length === 0) {
      return;
    }

    const publisherMap = new Map();
    for (const channel of channels) {
      for (const publisher of channel.publishers) {
        const list = publisherMap.get(publisher) || [];
        list.push(channel);
        publisherMap.set(publisher, list);
      }
    }

    let mempool;
    try {
      mempool = await this._withWalletOperation(() =>
        this.walletClient.getAddressMempoolEntries([...publisherMap.keys()])
      );
    } catch (error) {
      this.logger.debug?.(
        `[kasia-bridge] Live broadcast poll skipped: ${error?.message || error}`
      );
      return;
    }

    for (const [publisher, subscribedChannels] of publisherMap.entries()) {
      const addressEntry = mempoolEntriesForAddress(mempool, publisher);
      for (const transaction of senderTransactionsFromAddressEntry(addressEntry)) {
        const txId = txIdFromTransaction(transaction);
        if (!txId || hasProcessedTx(this.state, txId)) {
          continue;
        }
        try {
          const parsed = parseBroadcastPayload(txPayloadFromTransaction(transaction));
          const channel = subscribedChannels.find(
            (entry) => entry.channel_name === parsed.channelName
          );
          if (!channel) {
            continue;
          }
          const liveObservedMs = this.nowFn();
          channel.last_live_poll_ms = liveObservedMs;
          channel.recent_messages.push({
            tx_id: txId,
            sender_address: publisher,
            content: parsed.message,
            observed_live_ms: liveObservedMs,
            block_time_ms: null,
          });
          channel.recent_messages = channel.recent_messages.slice(-25);
          touchBroadcastChannel(channel, new Date(liveObservedMs).toISOString());
          this._enqueueEvent({
            eventType: "broadcast",
            messageId: txId,
            chatId: channel.channel_id,
            senderId: publisher,
            senderName: shortenAddress(publisher),
            body: parsed.message,
            timestampMs: liveObservedMs,
            channelName: channel.channel_name,
            raw: {
              transactionId: txId,
              delivery: publicDelivery(null, {
                liveObservedMs,
              }),
            },
          });
          markProcessedTx(this.state, txId, this.processedTxLimit);
        } catch (error) {
          this.logger.debug?.(
            `[kasia-bridge] Live broadcast tx ${txId} ignored: ${error?.message || error}`
          );
        }
      }
    }
  }

  async _pollOutboundHandshakeVisibility() {
    const initiatedConversations = Object.values(this.state.conversations || {}).filter(
      (conversation) =>
        conversation.pending_outbound_handshake?.tx_id &&
        conversation.peer_address
    );
    if (initiatedConversations.length === 0) {
      return;
    }

    for (const conversation of initiatedConversations) {
      const records = await this._fetchJson("/handshakes/by-sender", {
        address: this.walletInfo.address,
        block_time: Math.max(
          0,
          Number(conversation.last_handshake_block_time || 0) - DEFAULT_LIVE_LOOKBACK_MS
        ),
        limit: this.pollLimit,
      });
      const matched = Array.isArray(records)
        ? records.find(
            (record) =>
              String(record?.tx_id || "").trim() ===
              conversation.pending_outbound_handshake?.tx_id
          )
        : null;
      if (matched) {
        conversation.last_handshake_block_time = Math.max(
          Number(conversation.last_handshake_block_time || 0),
          txBlockTimeFromRecord(matched)
        );
      }
    }
  }

  async _pollSendJobLiveVisibility() {
    if (typeof this.walletClient.getAddressMempoolEntries !== "function") {
      return;
    }
    const pendingJobs = Object.values(this.state.send_jobs || {}).filter(
      (job) =>
        isIndexerTrackedSendJobStatus(job.status) &&
        Array.isArray(job.tx_ids) &&
        job.tx_ids.length > 0 &&
        Number(job.observed_live_ms || 0) <= 0
    );
    if (pendingJobs.length === 0) {
      return;
    }

    let mempool;
    try {
      mempool = await this._withWalletOperation(() =>
        this.walletClient.getAddressMempoolEntries([this.walletInfo.address])
      );
    } catch {
      return;
    }

    const addressEntry = mempoolEntriesForAddress(mempool, this.walletInfo.address);
    const liveTxIds = new Set(
      senderTransactionsFromAddressEntry(addressEntry)
        .map((transaction) => txIdFromTransaction(transaction))
        .filter(Boolean)
    );

    for (const job of pendingJobs) {
      if (job.tx_ids.some((txId) => liveTxIds.has(txId))) {
        const liveObservedMs = job.observed_live_ms || this.nowFn();
        await this._updateSendJob(job.job_id, {
          observed_live_ms: liveObservedMs,
          status:
            job.job_kind === "broadcast" ? "processed" : "waiting_for_indexer",
          indexed_ms: job.job_kind === "broadcast" ? liveObservedMs : job.indexed_ms,
          finished_ms: job.job_kind === "broadcast" ? liveObservedMs : job.finished_ms,
          indexed_parts:
            job.job_kind === "broadcast"
              ? job.total_parts || job.completed_parts || 1
              : job.indexed_parts,
          indexed_tx_ids:
            job.job_kind === "broadcast" ? [...job.tx_ids] : job.indexed_tx_ids,
        });
      }
    }
  }

  async syncOnce() {
    await this._pollLiveContextualMessages();
    await this._pollLiveBroadcastMessages();
    await this._pollHandshakes();
    await this._pollOutboundHandshakeVisibility();
    await this._pollContextualMessages();
    await this._pollSendJobLiveVisibility();
    await this._pollSendJobVisibility();
    await this._refreshConversationIdentities({ force: false });
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
        const conversation = this.state.conversations[String(record.sender || "").trim()];
        if (conversation) {
          await this._refreshConversationIdentity(conversation, { force: false });
        }
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
      conversation.pending_outbound_handshake = null;
      return;
    }

    conversation.their_alias = payload.alias;
    conversation.our_alias =
      conversation.our_alias || generateAlias(this.randomBytesFn);
    conversation.status =
      conversation.pending_outbound_handshake != null ? "awaiting_response" : "pending";
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
      senderName:
        conversation.display_name ||
        conversation.nickname ||
        conversation.kns_name ||
        shortenAddress(peerAddress),
      body: "Handshake request",
      timestampMs: Number(record.block_time || this.nowFn()),
      raw: {
        ...record,
        payload,
        delivery: publicDelivery(record, {
          indexedMs: this.nowFn(),
          indexedBlockTimeMs: Number(record.block_time || 0),
        }),
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
            senderName:
              conversation.display_name ||
              conversation.nickname ||
              conversation.kns_name ||
              shortenAddress(conversation.peer_address),
            body: messageBody,
            timestampMs: Number(record.block_time || this.nowFn()),
            raw: {
              ...record,
              delivery: publicDelivery(record, {
                indexedMs: this.nowFn(),
                indexedBlockTimeMs: Number(record.block_time || 0),
              }),
            },
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

  async _pollSendJobVisibility() {
    const pendingJobs = Object.values(this.state.send_jobs || {}).filter(
      (job) =>
        job.job_kind !== "broadcast" &&
        isIndexerTrackedSendJobStatus(job.status) &&
        Array.isArray(job.tx_ids) &&
        job.tx_ids.length > 0
    );

    for (const job of pendingJobs) {
      const conversation = this.state.conversations[String(job.chat_id || "").trim()];
      if (!conversation?.our_alias) {
        continue;
      }

      const trackedTxIds = new Set(job.tx_ids);
      const lookbackBlockTime = Math.max(
        0,
        Number(job.submitted_ms || job.started_ms || job.created_ms || 0) -
          DEFAULT_SEND_JOB_INDEXER_LOOKBACK_MS
      );
      const records = await this._fetchJson("/contextual-messages/by-sender", {
        address: this.walletInfo.address,
        alias: encodeIndexerAlias(conversation.our_alias),
        block_time: lookbackBlockTime,
        limit: Math.max(this.pollLimit, trackedTxIds.size * 10),
      });

      const matchedRecords = Array.isArray(records)
        ? records.filter((record) => trackedTxIds.has(record?.tx_id))
        : [];
      const indexedTxIds = job.tx_ids.filter((txId) =>
        matchedRecords.some((record) => String(record?.tx_id || "").trim() === txId)
      );
      const indexedParts = indexedTxIds.length;
      const indexedBlockTimeMs = matchedRecords.reduce(
        (maxBlockTime, record) =>
          Math.max(maxBlockTime, Number(record?.block_time || 0)),
        Number(job.indexed_block_time_ms || 0)
      );

      const updates = {
        indexed_parts: indexedParts,
        indexed_tx_ids: indexedTxIds,
        indexed_block_time_ms: indexedBlockTimeMs || null,
      };

      if (indexedParts >= trackedTxIds.size) {
        updates.status = "processed";
        updates.indexed_ms = job.indexed_ms ?? this.nowFn();
        updates.finished_ms = this.nowFn();
      } else if (job.status !== "waiting_for_indexer") {
        updates.status = "waiting_for_indexer";
      } else {
        updates.status = "waiting_for_indexer";
      }

      const hasChanged =
        updates.status !== job.status ||
        updates.indexed_parts !== Number(job.indexed_parts || 0) ||
        JSON.stringify(updates.indexed_tx_ids) !==
          JSON.stringify(job.indexed_tx_ids || []) ||
        updates.indexed_block_time_ms !==
          (job.indexed_block_time_ms ?? null) ||
        updates.indexed_ms !== (job.indexed_ms ?? null) ||
        updates.finished_ms !== (job.finished_ms ?? null);

      if (hasChanged) {
        await this._updateSendJob(job.job_id, updates);
      }
    }
  }

  async _fetchJson(path, params) {
    const candidates = this.indexerPool.getCandidates();
    let lastError = null;
    for (const indexerUrl of candidates) {
      const url = joinUrl(indexerUrl, path, params);
      try {
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

        this.indexerPool.markSuccess(indexerUrl);
        return await response.json();
      } catch (error) {
        lastError = error;
        this.indexerPool.markFailure(indexerUrl, error?.message || error);
      }
    }

    throw lastError || new Error(`No Kasia indexer endpoint could satisfy ${path}`);
  }

  _enqueueEvent(event) {
    event.raw = {
      ...(event.raw || {}),
      delivery: publicDelivery(event.raw, {
        deliveredToHermesMs: this.nowFn(),
      }),
    };
    this.messageQueue.push(event);
    if (this.messageQueue.length > this.maxQueueSize) {
      this.messageQueue.splice(0, this.messageQueue.length - this.maxQueueSize);
    }
  }

  async _saveState() {
    this.state.wallet.send_state = this.walletClient.exportSendState?.() || {};
    const snapshot = JSON.parse(JSON.stringify(this.state));
    const write = this._saveStateTail
      .catch(() => {})
      .then(() => saveState(this.statePath, snapshot));
    this._saveStateTail = write;
    await write;
  }
}
