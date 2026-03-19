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

const DEFAULT_CONTEXTUAL_MESSAGE_MAX_CHARS = 240;

function isPayloadTooLargeError(error) {
  const text = String(error?.message || error || "").toLowerCase();
  return (
    text.includes("storage mass") ||
    text.includes("larger than max allowed size") ||
    text.includes("transaction is not standard")
  );
}

function truncateMessage(content, maxLength = DEFAULT_CONTEXTUAL_MESSAGE_MAX_CHARS) {
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

  if (chunks.length > 1) {
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
    await this.walletClient.hydrateSendState?.();
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

    const text = String(message);
    let txResults;
    try {
      txResults = [await this._sendConversationMessageChunk(conversation, text)];
    } catch (error) {
      if (!isPayloadTooLargeError(error)) {
        throw error;
      }
      const chunks = truncateMessage(text);
      if (chunks.length <= 1) {
        throw error;
      }
      txResults = [];
      for (const chunk of chunks) {
        txResults.push(
          await this._sendConversationMessageChunk(conversation, chunk)
        );
      }
    }

    touchConversation(conversation, new Date(this.nowFn()).toISOString());
    await this._saveState();

    const txIds = txResults.map((result) => result?.txId || result).filter(Boolean);
    const lastResult = txResults[txResults.length - 1];
    return {
      status: "sent",
      txId: txIds[txIds.length - 1],
      txIds,
      partCount: txResults.length,
      chatId: conversation.peer_address,
      wallet: typeof lastResult === "object" ? lastResult : undefined,
    };
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
