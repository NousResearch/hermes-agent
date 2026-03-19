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
  }

  async init() {
    this.walletInfo = await this.walletClient.init();
    this.state = await loadState(this.statePath, {
      address: this.walletInfo.address,
      public_key: this.walletInfo.publicKeyHex,
      network: this.walletInfo.network,
    });
    this.walletClient.loadSendState?.(this.state.wallet.send_state || {});
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
    await this._saveState();
    await this.walletClient.close?.();
  }

  health() {
    const sendState = this.walletClient.exportSendState?.() || {};
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

  async send({ chatId, message }) {
    if (!message || !String(message).trim()) {
      throw new Error("Message is required");
    }

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

    const payloadBytes = buildContextualMessageTransactionPayload({
      recipientAddress: conversation.peer_address,
      alias: conversation.our_alias,
      message: String(message),
      randomBytesFn: this.randomBytesFn,
    });
    const txResult = await this.walletClient.sendPayloadTransaction({
      destinationAddress: this.walletInfo.address,
      amountSompi: MINIMUM_MESSAGE_AMOUNT_SOMPI,
      payloadBytes,
      strategy: "contextual",
    });

    touchConversation(conversation, new Date(this.nowFn()).toISOString());
    await this._saveState();

    return {
      status: "sent",
      txId: txResult.txId || txResult,
      chatId: conversation.peer_address,
      wallet: typeof txResult === "object" ? txResult : undefined,
    };
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
      }
      markProcessedTx(this.state, record.tx_id, this.processedTxLimit);
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
