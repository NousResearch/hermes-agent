import { Address, XOnlyPublicKey, kaspaToSompi } from "./kaspa_sdk.js";
import {
  createCipheriv,
  createDecipheriv,
  createECDH,
  hkdfSync,
  randomBytes,
} from "node:crypto";

export const PROTOCOL_PREFIX = "ciph_msg:";
export const HANDSHAKE_PREFIX = "ciph_msg:1:handshake:";
export const CONTEXTUAL_PREFIX = "ciph_msg:1:comm:";
export const BROADCAST_PREFIX = "ciph_msg:1:bcast:";
export const ALIAS_LENGTH_BYTES = 6;
export const MINIMUM_MESSAGE_AMOUNT_SOMPI = kaspaToSompi("0.2");

function hkdfKey(sharedSecret) {
  return Buffer.from(
    hkdfSync(
      "sha256",
      Buffer.from(sharedSecret),
      Buffer.alloc(0),
      Buffer.alloc(0),
      32
    )
  );
}

function compressedPublicKeyForAddress(address) {
  const xOnlyHex = XOnlyPublicKey.fromAddress(new Address(address)).toString();
  return Buffer.concat([Buffer.from([0x02]), Buffer.from(xOnlyHex, "hex")]);
}

export function generateAlias(randomBytesFn = randomBytes) {
  return randomBytesFn(ALIAS_LENGTH_BYTES).toString("hex");
}

export function ensureAlias(alias) {
  if (typeof alias !== "string" || alias.length !== ALIAS_LENGTH_BYTES * 2) {
    throw new Error("Alias must be a 12-character hex string");
  }
  if (!/^[0-9a-f]+$/i.test(alias)) {
    throw new Error("Alias must be hex");
  }
}

export function encryptToAddress(
  address,
  plaintext,
  randomBytesFn = randomBytes
) {
  const receiverPublicKey = compressedPublicKeyForAddress(address);
  const ecdh = createECDH("secp256k1");
  ecdh.generateKeys();

  const sharedSecret = ecdh.computeSecret(receiverPublicKey);
  const symmetricKey = hkdfKey(sharedSecret);
  const nonce = randomBytesFn(12);
  const cipher = createCipheriv("chacha20-poly1305", symmetricKey, nonce, {
    authTagLength: 16,
  });
  const ciphertext = Buffer.concat([
    cipher.update(Buffer.from(String(plaintext), "utf8")),
    cipher.final(),
  ]);
  const authTag = cipher.getAuthTag();

  return Buffer.concat([
    nonce,
    ecdh.getPublicKey(undefined, "compressed"),
    ciphertext,
    authTag,
  ]);
}

export function decryptSealedMessage(privateKeyHex, sealedMessage) {
  const sealedBuffer = Buffer.isBuffer(sealedMessage)
    ? sealedMessage
    : Buffer.from(String(sealedMessage), "hex");

  if (sealedBuffer.length < 61) {
    throw new Error("Encrypted payload is too short");
  }

  const nonce = sealedBuffer.subarray(0, 12);
  const ephemeralPublicKey = sealedBuffer.subarray(12, 45);
  const ciphertext = sealedBuffer.subarray(45, -16);
  const authTag = sealedBuffer.subarray(-16);

  const ecdh = createECDH("secp256k1");
  ecdh.setPrivateKey(Buffer.from(privateKeyHex, "hex"));
  const sharedSecret = ecdh.computeSecret(ephemeralPublicKey);
  const symmetricKey = hkdfKey(sharedSecret);

  const decipher = createDecipheriv(
    "chacha20-poly1305",
    symmetricKey,
    nonce,
    { authTagLength: 16 }
  );
  decipher.setAuthTag(authTag);

  return Buffer.concat([
    decipher.update(ciphertext),
    decipher.final(),
  ]).toString("utf8");
}

export function buildHandshakePayload({
  alias,
  theirAlias,
  isResponse = false,
  timestamp = Date.now(),
  version = 1,
}) {
  ensureAlias(alias);
  if (theirAlias != null) {
    ensureAlias(theirAlias);
  }

  return {
    type: "handshake",
    alias,
    ...(theirAlias ? { theirAlias } : {}),
    timestamp,
    version,
    ...(isResponse ? { isResponse: true } : {}),
  };
}

export function parseHandshakePayload(plaintext) {
  const parsed = JSON.parse(plaintext);
  if (parsed?.type !== "handshake") {
    throw new Error("Unsupported handshake payload");
  }
  ensureAlias(parsed.alias);
  if (parsed.theirAlias != null) {
    ensureAlias(parsed.theirAlias);
  }
  return parsed;
}

export function buildHandshakeTransactionPayload({
  recipientAddress,
  payload,
  randomBytesFn = randomBytes,
}) {
  const sealed = encryptToAddress(
    recipientAddress,
    JSON.stringify(payload),
    randomBytesFn
  );
  return Buffer.concat([Buffer.from(HANDSHAKE_PREFIX, "utf8"), sealed]);
}

function looksLikeBase64(value) {
  return /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}(?:==)?|[A-Za-z0-9+/]{3}=?)?$/.test(
    value
  );
}

export function decodeIndexedContextualMessagePayload(messagePayloadHex) {
  const asciiCandidate = Buffer.from(messagePayloadHex, "hex").toString("utf8");
  if (asciiCandidate.length > 2 && looksLikeBase64(asciiCandidate)) {
    return Buffer.from(asciiCandidate, "base64");
  }
  return Buffer.from(messagePayloadHex, "hex");
}

export function buildContextualMessageTransactionPayload({
  recipientAddress,
  alias,
  message,
  randomBytesFn = randomBytes,
}) {
  ensureAlias(alias);
  const sealed = encryptToAddress(recipientAddress, message, randomBytesFn);
  const base64Payload = sealed.toString("base64");
  return Buffer.from(
    `${CONTEXTUAL_PREFIX}${alias}:${base64Payload}`,
    "utf8"
  );
}

export function normalizeBroadcastChannelName(channelName) {
  const normalized = String(channelName || "").trim().toLowerCase();
  if (!normalized) {
    throw new Error("Broadcast channel name is required");
  }
  if (!/^[a-z0-9][a-z0-9_-]{0,31}$/i.test(normalized)) {
    throw new Error(
      "Broadcast channel names must start with a letter or number and only use letters, numbers, hyphen, or underscore"
    );
  }
  return normalized;
}

export function buildBroadcastTransactionPayload({ channelName, message }) {
  const normalizedChannel = normalizeBroadcastChannelName(channelName);
  const text = String(message || "").trim();
  if (!text) {
    throw new Error("Broadcast message is required");
  }
  return Buffer.from(
    `${BROADCAST_PREFIX}${normalizedChannel}:${text}`,
    "utf8"
  );
}

function payloadBuffer(payload) {
  if (Buffer.isBuffer(payload)) {
    return payload;
  }
  if (payload instanceof Uint8Array) {
    return Buffer.from(payload);
  }
  const text = String(payload || "").trim();
  if (!text) {
    return Buffer.alloc(0);
  }
  if (/^(?:[0-9a-f]{2})+$/i.test(text)) {
    return Buffer.from(text, "hex");
  }
  return Buffer.from(text, "utf8");
}

function payloadString(payload) {
  return payloadBuffer(payload).toString("utf8");
}

export function parseContextualMessagePayload(payload) {
  const text = payloadString(payload);
  if (!text.startsWith(CONTEXTUAL_PREFIX)) {
    throw new Error("Unsupported contextual payload");
  }
  const remainder = text.slice(CONTEXTUAL_PREFIX.length);
  const delimiterIndex = remainder.indexOf(":");
  if (delimiterIndex <= 0) {
    throw new Error("Contextual payload is missing an alias delimiter");
  }
  const alias = remainder.slice(0, delimiterIndex);
  const base64Payload = remainder.slice(delimiterIndex + 1);
  ensureAlias(alias);
  if (!base64Payload) {
    throw new Error("Contextual payload is missing encrypted content");
  }
  return {
    alias,
    sealedMessage: Buffer.from(base64Payload, "base64"),
  };
}

export function parseBroadcastPayload(payload) {
  const text = payloadString(payload);
  if (!text.startsWith(BROADCAST_PREFIX)) {
    throw new Error("Unsupported broadcast payload");
  }
  const remainder = text.slice(BROADCAST_PREFIX.length);
  const delimiterIndex = remainder.indexOf(":");
  if (delimiterIndex <= 0) {
    throw new Error("Broadcast payload is missing a channel delimiter");
  }
  const channelName = normalizeBroadcastChannelName(
    remainder.slice(0, delimiterIndex)
  );
  const message = remainder.slice(delimiterIndex + 1);
  if (!message.trim()) {
    throw new Error("Broadcast payload is missing message content");
  }
  return {
    channelName,
    message,
  };
}

export function shortenAddress(address) {
  const value = String(address || "");
  if (value.length <= 24) {
    return value;
  }
  return `${value.slice(0, 14)}...${value.slice(-8)}`;
}
