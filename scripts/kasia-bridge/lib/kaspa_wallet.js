import { setTimeout as delay } from "node:timers/promises";

import {
  Address,
  addressFromScriptPublicKey,
  calculateTransactionFee,
  ConnectStrategy,
  createTransaction,
  Encoding,
  Generator,
  Mnemonic,
  NetworkId,
  PaymentOutput,
  PrivateKeyGenerator,
  RpcClient,
  signTransaction,
  TransactionOutput,
  UtxoContext,
  UtxoEntries,
  UtxoProcessor,
  updateTransactionMass,
  XPrv,
  payToAddressScript,
} from "./kaspa_sdk.js";

const DEFAULT_SEND_STATE = Object.freeze({
  reserved_outpoints: [],
  pending_outputs: [],
  last_compaction_ms: 0,
});
const DEFAULT_SEND_VISIBILITY_RETRIES = 10;
const DEFAULT_SEND_VISIBILITY_DELAY_MS = 1000;
const DEFAULT_MAX_CONFIRMED_INPUT_PLANS = 6;
const DEFAULT_COMPACTION_INPUT_THRESHOLD = 3;
const DEFAULT_COMPACTION_MAX_INPUTS = 12;
const DEFAULT_COMPACTION_COOLDOWN_MS = 60_000;
const DEFAULT_RESERVED_OUTPOINT_TTL_MS = 120_000;
const DEFAULT_PENDING_OUTPUT_TTL_MS = 600_000;
const DUST_THRESHOLD_SOMPI = 10_000n;
const DEFAULT_LOCAL_PENDING_RETENTION_MS = 30_000;
const SYNTHETIC_PREVIEW_INPUT_AMOUNT_SOMPI = 10_000_000_000n;
const DEFAULT_FEE_POLICY = "priority";
const DEFAULT_FEE_ESTIMATE_TTL_MS = 15_000;
const DEFAULT_AUTO_FEE_TARGET_SECONDS = 8;
const DEFAULT_LOW_FEE_RATE_SOMPI_PER_GRAM = 1;
const DEFAULT_NORMAL_FEE_RATE_SOMPI_PER_GRAM = 2;
const DEFAULT_PRIORITY_FEE_RATE_SOMPI_PER_GRAM = 6;

function toBigInt(value, fallback = 0n) {
  if (typeof value === "bigint") {
    return value;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return BigInt(Math.trunc(value));
  }
  if (typeof value === "string" && value.trim()) {
    try {
      return BigInt(value.trim());
    } catch {}
  }
  return fallback;
}

function toNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function toPositiveNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : null;
}

export function normalizeFeePolicy(value, fallback = DEFAULT_FEE_POLICY) {
  const normalized = String(value || "").trim().toLowerCase();
  if (
    normalized === "low" ||
    normalized === "normal" ||
    normalized === "priority" ||
    normalized === "auto"
  ) {
    return normalized;
  }
  return normalizeFeePolicy(
    fallback === value ? DEFAULT_FEE_POLICY : fallback,
    DEFAULT_FEE_POLICY
  );
}

function getFallbackFeeRate(policy = DEFAULT_FEE_POLICY) {
  switch (normalizeFeePolicy(policy)) {
    case "low":
      return DEFAULT_LOW_FEE_RATE_SOMPI_PER_GRAM;
    case "normal":
    case "auto":
      return DEFAULT_NORMAL_FEE_RATE_SOMPI_PER_GRAM;
    case "priority":
    default:
      return DEFAULT_PRIORITY_FEE_RATE_SOMPI_PER_GRAM;
  }
}

function getFirstPositiveBucket(buckets = []) {
  if (!Array.isArray(buckets)) {
    return null;
  }
  return buckets.find((bucket) => toPositiveNumber(bucket?.feerate) != null) || null;
}

function getLastPositiveBucket(buckets = []) {
  if (!Array.isArray(buckets)) {
    return null;
  }
  for (let index = buckets.length - 1; index >= 0; index -= 1) {
    if (toPositiveNumber(buckets[index]?.feerate) != null) {
      return buckets[index];
    }
  }
  return null;
}

export function selectFeeRateFromEstimate(
  feeEstimate,
  policy = DEFAULT_FEE_POLICY,
  { autoTargetSeconds = DEFAULT_AUTO_FEE_TARGET_SECONDS } = {}
) {
  const estimate =
    feeEstimate && typeof feeEstimate.estimate === "object"
      ? feeEstimate.estimate
      : feeEstimate || {};
  const priorityRate = toPositiveNumber(estimate?.priorityBucket?.feerate);
  const firstNormalBucket = getFirstPositiveBucket(estimate?.normalBuckets);
  const lastNormalBucket = getLastPositiveBucket(estimate?.normalBuckets);
  const lowBucket =
    getFirstPositiveBucket(estimate?.lowBuckets) || lastNormalBucket;
  const normalRate = toPositiveNumber(firstNormalBucket?.feerate);
  const lowRate = toPositiveNumber(lowBucket?.feerate);

  switch (normalizeFeePolicy(policy)) {
    case "low":
      return lowRate || normalRate || priorityRate || null;
    case "normal":
      return normalRate || priorityRate || lowRate || null;
    case "auto": {
      const normalSeconds = toPositiveNumber(firstNormalBucket?.estimatedSeconds);
      if (normalRate && (normalSeconds == null || normalSeconds <= autoTargetSeconds)) {
        return normalRate;
      }
      return priorityRate || normalRate || lowRate || null;
    }
    case "priority":
    default:
      return priorityRate || normalRate || lowRate || null;
  }
}

function normalizeOutpoint(outpoint = {}) {
  const transactionId =
    outpoint.transactionId ||
    outpoint.transaction_id ||
    outpoint.txId ||
    outpoint.tx_id ||
    null;
  const index = toNumber(outpoint.index, -1);
  return {
    transactionId,
    index,
  };
}

export function makeOutpointKey(inputOrTransactionId, index) {
  if (
    inputOrTransactionId &&
    typeof inputOrTransactionId === "object" &&
    inputOrTransactionId.outpoint
  ) {
    const normalized = normalizeOutpoint(inputOrTransactionId.outpoint);
    return makeOutpointKey(normalized.transactionId, normalized.index);
  }

  const txId = String(inputOrTransactionId || "").trim();
  const outputIndex = toNumber(index, -1);
  if (!txId || outputIndex < 0) {
    return null;
  }
  return `${txId}:${outputIndex}`;
}

function normalizeReservedOutpoint(entry = {}) {
  const key = String(entry.key || entry.outpoint_key || "").trim();
  if (!key) {
    return null;
  }
  return {
    key,
    reserved_at_ms: toNumber(entry.reserved_at_ms, 0),
  };
}

function normalizePendingOutput(entry = {}) {
  const key =
    String(entry.key || "").trim() ||
    makeOutpointKey(entry.tx_id || entry.transaction_id, entry.index);
  if (!key) {
    return null;
  }
  return {
    key,
    tx_id: String(entry.tx_id || entry.transaction_id || "").trim(),
    index: toNumber(entry.index, 0),
    amount: String(toBigInt(entry.amount, 0n)),
    created_ms: toNumber(entry.created_ms, 0),
    observed_in_mempool: Boolean(
      entry.observed_in_mempool ?? entry.observedInMempool
    ),
  };
}

function outputAddress(output, networkId) {
  try {
    return normalizeAddressValue(
      addressFromScriptPublicKey(output?.scriptPublicKey, networkId)
    );
  } catch {
    return output?.verboseData?.scriptPublicKeyAddress || null;
  }
}

function mempoolEntriesForAddress(mempoolResponse, address) {
  const entries = Array.isArray(mempoolResponse?.entries)
    ? mempoolResponse.entries
    : [];
  return entries.find((entry) => entry?.address === address) || null;
}

function pendingOutputsFromMempoolEntry(mempoolEntry, address, networkId, nowMs) {
  const transaction = mempoolEntry?.transaction;
  if (!transaction || !Array.isArray(transaction.outputs)) {
    return [];
  }

  return transaction.outputs
    .map((output, index) => {
      if (outputAddress(output, networkId) !== address) {
        return null;
      }
      const txId = normalizeAddressValue(transaction?.verboseData?.transactionId);
      const key = makeOutpointKey(txId, index);
      if (!key) {
        return null;
      }
      return {
        key,
        tx_id: txId,
        index,
        amount: String(toBigInt(output?.value, 0n)),
        created_ms: nowMs,
        observed_in_mempool: true,
      };
    })
    .filter(Boolean);
}

export function normalizeSendState(sendState = {}) {
  const reserved = Array.isArray(sendState.reserved_outpoints)
    ? sendState.reserved_outpoints
        .map((entry) => normalizeReservedOutpoint(entry))
        .filter(Boolean)
    : [];
  const pending = Array.isArray(sendState.pending_outputs)
    ? sendState.pending_outputs
        .map((entry) => normalizePendingOutput(entry))
        .filter(Boolean)
    : [];

  return {
    reserved_outpoints: reserved,
    pending_outputs: pending,
    last_compaction_ms: toNumber(sendState.last_compaction_ms, 0),
  };
}

export function normalizeUtxoList(entries) {
  if (!entries) {
    return [];
  }
  if (Array.isArray(entries)) {
    return entries.filter(Boolean);
  }
  if (Array.isArray(entries.items)) {
    return entries.items.filter(Boolean);
  }
  if (typeof entries[Symbol.iterator] === "function") {
    return Array.from(entries).filter(Boolean);
  }
  return [];
}

function isPendingUtxo(entry) {
  return toBigInt(entry?.blockDaaScore, 0n) === 0n;
}

function isSpendableUtxo(entry) {
  return Boolean(entry) && entry.isCoinbase !== true;
}

function sortByAmountDescending(entries) {
  return [...entries].sort((left, right) => {
    const diff = toBigInt(right?.amount, 0n) - toBigInt(left?.amount, 0n);
    if (diff === 0n) {
      return 0;
    }
    return diff > 0n ? 1 : -1;
  });
}

function getTrackedPendingKeys(sendState) {
  return new Set(sendState.pending_outputs.map((entry) => entry.key));
}

export function reconcileSendState(
  sendState,
  liveUtxos,
  {
    nowMs = Date.now(),
    reservedOutpointTtlMs = DEFAULT_RESERVED_OUTPOINT_TTL_MS,
    pendingOutputTtlMs = DEFAULT_PENDING_OUTPUT_TTL_MS,
  } = {}
) {
  const normalized = normalizeSendState(sendState);
  const liveEntries = normalizeUtxoList(liveUtxos);
  const liveByKey = new Map();

  for (const entry of liveEntries) {
    const key = makeOutpointKey(entry);
    if (key) {
      liveByKey.set(key, entry);
    }
  }

  normalized.reserved_outpoints = normalized.reserved_outpoints.filter(
    (entry) => entry.key && entry.reserved_at_ms + reservedOutpointTtlMs > nowMs
  );
  const reservedKeys = new Set(
    normalized.reserved_outpoints.map((entry) => entry.key)
  );

  normalized.pending_outputs = normalized.pending_outputs.filter((entry) => {
    if (!entry.key) {
      return false;
    }
    if (reservedKeys.has(entry.key)) {
      return false;
    }
    const live = liveByKey.get(entry.key);
    if (!live) {
      return entry.created_ms + pendingOutputTtlMs > nowMs;
    }
    return isPendingUtxo(live);
  });

  return normalized;
}

export function buildCandidateUtxoPlans({
  trackedPendingUtxos = [],
  matureUtxos = [],
  maxConfirmedPlans = DEFAULT_MAX_CONFIRMED_INPUT_PLANS,
}) {
  const plans = [];
  const seen = new Set();

  for (const utxo of sortByAmountDescending(trackedPendingUtxos)) {
    const key = makeOutpointKey(utxo);
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    plans.push({
      name: "pending-single",
      entries: [utxo],
      usesPendingInputs: true,
    });
  }

  const confirmed = sortByAmountDescending(matureUtxos);
  const planCount = Math.min(confirmed.length, Math.max(1, maxConfirmedPlans));
  for (let count = 1; count <= planCount; count += 1) {
    plans.push({
      name: `confirmed-${count}`,
      entries: confirmed.slice(0, count),
      usesPendingInputs: false,
    });
  }

  if (confirmed.length > planCount) {
    plans.push({
      name: "confirmed-all",
      entries: confirmed,
      usesPendingInputs: false,
    });
  }

  return plans;
}

export function shouldCompactSend({
  matureUtxos = [],
  trackedPendingUtxos = [],
  lastCompactionMs = 0,
  nowMs = Date.now(),
  cooldownMs = DEFAULT_COMPACTION_COOLDOWN_MS,
  threshold = DEFAULT_COMPACTION_INPUT_THRESHOLD,
}) {
  if (trackedPendingUtxos.length > 0) {
    return false;
  }
  if (matureUtxos.length < threshold) {
    return false;
  }
  return nowMs - toNumber(lastCompactionMs, 0) >= cooldownMs;
}

function sumUtxoAmounts(entries) {
  return normalizeUtxoList(entries).reduce(
    (total, entry) => total + toBigInt(entry?.amount, 0n),
    0n
  );
}

function dedupeUtxos(entries) {
  const byKey = new Map();
  for (const entry of normalizeUtxoList(entries)) {
    const key = makeOutpointKey(entry);
    if (!key) {
      continue;
    }
    byKey.set(key, entry);
  }
  return [...byKey.values()];
}

function compareContextualUtxos(left, right) {
  const amountDiff = toBigInt(right?.amount, 0n) - toBigInt(left?.amount, 0n);
  if (amountDiff !== 0n) {
    return amountDiff > 0n ? 1 : -1;
  }

  const leftPending = isPendingUtxo(left);
  const rightPending = isPendingUtxo(right);
  if (leftPending !== rightPending) {
    return leftPending ? -1 : 1;
  }

  const leftKey = makeOutpointKey(left) || "";
  const rightKey = makeOutpointKey(right) || "";
  return rightKey.localeCompare(leftKey);
}

export function previewRawSelfSpend({
  entries,
  payloadBytes,
  networkId,
  scriptPublicKey,
  feeRateSompiPerGram = DEFAULT_LOW_FEE_RATE_SOMPI_PER_GRAM,
  dustThresholdSompi = DUST_THRESHOLD_SOMPI,
}) {
  const normalizedEntries = normalizeUtxoList(entries);
  if (normalizedEntries.length === 0) {
    throw new Error("No spendable UTXOs selected");
  }

  const totalInput = sumUtxoAmounts(normalizedEntries);
  if (totalInput <= 0n) {
    throw new Error("Selected UTXOs do not carry a spendable amount");
  }

  const transaction = createTransaction(
    normalizedEntries,
    [],
    0n,
    payloadBytes,
    1
  );
  transaction.outputs = [new TransactionOutput(totalInput, scriptPublicKey)];

  let fee = calculateTransactionFee(
    networkId,
    transaction,
    feeRateSompiPerGram
  );
  let outputAmount = totalInput - toBigInt(fee, 0n);
  if (outputAmount <= dustThresholdSompi) {
    throw new Error("Insufficient funds after fee");
  }

  transaction.outputs = [new TransactionOutput(outputAmount, scriptPublicKey)];
  if (!updateTransactionMass(networkId, transaction, 1)) {
    throw new Error("Transaction is not standard: storage mass too large");
  }

  fee = calculateTransactionFee(networkId, transaction, feeRateSompiPerGram);
  outputAmount = totalInput - toBigInt(fee, 0n);
  if (outputAmount <= dustThresholdSompi) {
    throw new Error("Insufficient funds after fee");
  }

  transaction.outputs = [new TransactionOutput(outputAmount, scriptPublicKey)];
  if (!updateTransactionMass(networkId, transaction, 1)) {
    throw new Error("Transaction is not standard: storage mass too large");
  }

  return {
    transaction,
    fee: toBigInt(fee, 0n),
    outputAmount,
    inputCount: normalizedEntries.length,
    usesPendingInputs: normalizedEntries.some(isPendingUtxo),
  };
}

export function buildRawSelfSpendTransaction({
  entries,
  payloadBytes,
  networkId,
  scriptPublicKey,
  privateKey,
  feeRateSompiPerGram = DEFAULT_LOW_FEE_RATE_SOMPI_PER_GRAM,
  dustThresholdSompi = DUST_THRESHOLD_SOMPI,
}) {
  const preview = previewRawSelfSpend({
    entries,
    payloadBytes,
    networkId,
    scriptPublicKey,
    feeRateSompiPerGram,
    dustThresholdSompi,
  });
  const signedTransaction = signTransaction(
    preview.transaction,
    [privateKey],
    false
  );
  return {
    ...preview,
    transaction: signedTransaction,
  };
}

export function previewRawDirectedSpend({
  entries,
  amountSompi,
  payloadBytes,
  networkId,
  destinationScriptPublicKey,
  changeScriptPublicKey,
  feeRateSompiPerGram = DEFAULT_LOW_FEE_RATE_SOMPI_PER_GRAM,
  dustThresholdSompi = DUST_THRESHOLD_SOMPI,
}) {
  const normalizedEntries = normalizeUtxoList(entries);
  if (normalizedEntries.length === 0) {
    throw new Error("No spendable UTXOs selected");
  }

  const spendAmount = toBigInt(amountSompi, 0n);
  if (spendAmount <= 0n) {
    throw new Error("Directed spends must specify a positive amount");
  }

  const totalInput = sumUtxoAmounts(normalizedEntries);
  if (totalInput <= spendAmount) {
    throw new Error("Insufficient funds after fee");
  }

  const transaction = createTransaction(
    normalizedEntries,
    [],
    0n,
    payloadBytes,
    1
  );

  let outputs = [
    new TransactionOutput(spendAmount, destinationScriptPublicKey),
    new TransactionOutput(totalInput - spendAmount, changeScriptPublicKey),
  ];
  transaction.outputs = outputs;
  let fee = calculateTransactionFee(
    networkId,
    transaction,
    feeRateSompiPerGram
  );
  let changeAmount = totalInput - spendAmount - toBigInt(fee, 0n);

  if (changeAmount > dustThresholdSompi) {
    outputs = [
      new TransactionOutput(spendAmount, destinationScriptPublicKey),
      new TransactionOutput(changeAmount, changeScriptPublicKey),
    ];
  } else {
    fee = calculateTransactionFee(
      networkId,
      Object.assign(transaction, {
        outputs: [new TransactionOutput(spendAmount, destinationScriptPublicKey)],
      }),
      feeRateSompiPerGram
    );
    changeAmount = totalInput - spendAmount - toBigInt(fee, 0n);
    outputs = [new TransactionOutput(spendAmount, destinationScriptPublicKey)];
  }

  if (changeAmount < 0n) {
    throw new Error("Insufficient funds after fee");
  }

  transaction.outputs = outputs;
  if (!updateTransactionMass(networkId, transaction, 1)) {
    throw new Error("Transaction is not standard: storage mass too large");
  }

  fee = calculateTransactionFee(networkId, transaction, feeRateSompiPerGram);
  if (outputs.length === 2) {
    changeAmount = totalInput - spendAmount - toBigInt(fee, 0n);
    if (changeAmount <= dustThresholdSompi) {
      outputs = [new TransactionOutput(spendAmount, destinationScriptPublicKey)];
    } else {
      outputs = [
        new TransactionOutput(spendAmount, destinationScriptPublicKey),
        new TransactionOutput(changeAmount, changeScriptPublicKey),
      ];
    }
  } else {
    changeAmount = totalInput - spendAmount - toBigInt(fee, 0n);
  }

  if (changeAmount < 0n) {
    throw new Error("Insufficient funds after fee");
  }

  transaction.outputs = outputs;
  if (!updateTransactionMass(networkId, transaction, 1)) {
    throw new Error("Transaction is not standard: storage mass too large");
  }

  return {
    transaction,
    fee: toBigInt(fee, 0n),
    changeAmount,
    inputCount: normalizedEntries.length,
    usesPendingInputs: normalizedEntries.some(isPendingUtxo),
  };
}

export function buildRawDirectedSpendTransaction({
  entries,
  amountSompi,
  payloadBytes,
  networkId,
  destinationScriptPublicKey,
  changeScriptPublicKey,
  privateKey,
  feeRateSompiPerGram = DEFAULT_LOW_FEE_RATE_SOMPI_PER_GRAM,
  dustThresholdSompi = DUST_THRESHOLD_SOMPI,
}) {
  const preview = previewRawDirectedSpend({
    entries,
    amountSompi,
    payloadBytes,
    networkId,
    destinationScriptPublicKey,
    changeScriptPublicKey,
    feeRateSompiPerGram,
    dustThresholdSompi,
  });
  const signedTransaction = signTransaction(
    preview.transaction,
    [privateKey],
    false
  );
  return {
    ...preview,
    transaction: signedTransaction,
  };
}

export function selectContextualRawEntries({
  availablePendingUtxos = [],
  trackedPendingUtxos = [],
  matureUtxos = [],
  payloadBytes,
  networkId,
  scriptPublicKey,
  feeRateSompiPerGram = DEFAULT_LOW_FEE_RATE_SOMPI_PER_GRAM,
  dustThresholdSompi = DUST_THRESHOLD_SOMPI,
}) {
  const trackedPending = sortByAmountDescending(
    dedupeUtxos(trackedPendingUtxos)
  );
  for (const entry of trackedPending) {
    try {
      previewRawSelfSpend({
        entries: [entry],
        payloadBytes,
        networkId,
        scriptPublicKey,
        feeRateSompiPerGram,
        dustThresholdSompi,
      });
      return [entry];
    } catch (error) {
      if (!isLikelyInsufficientFundsError(error)) {
        throw error;
      }
    }
  }

  const prioritized = dedupeUtxos([
    ...normalizeUtxoList(availablePendingUtxos),
    ...normalizeUtxoList(matureUtxos),
  ]).sort(compareContextualUtxos);

  if (prioritized.length === 0) {
    throw new Error("Insufficient funds after fee");
  }

  const selected = [];
  for (const entry of prioritized) {
    selected.push(entry);
    try {
      previewRawSelfSpend({
        entries: selected,
        payloadBytes,
        networkId,
        scriptPublicKey,
        feeRateSompiPerGram,
        dustThresholdSompi,
      });
      return [...selected];
    } catch (error) {
      if (!isLikelyInsufficientFundsError(error)) {
        throw error;
      }
    }
  }

  throw new Error("Insufficient funds after fee");
}

export function selectDirectedRawEntries({
  availablePendingUtxos = [],
  trackedPendingUtxos = [],
  matureUtxos = [],
  amountSompi,
  payloadBytes,
  networkId,
  destinationScriptPublicKey,
  changeScriptPublicKey,
  feeRateSompiPerGram = DEFAULT_LOW_FEE_RATE_SOMPI_PER_GRAM,
  dustThresholdSompi = DUST_THRESHOLD_SOMPI,
}) {
  const trackedPending = sortByAmountDescending(
    dedupeUtxos(trackedPendingUtxos)
  );
  for (const entry of trackedPending) {
    try {
      previewRawDirectedSpend({
        entries: [entry],
        amountSompi,
        payloadBytes,
        networkId,
        destinationScriptPublicKey,
        changeScriptPublicKey,
        feeRateSompiPerGram,
        dustThresholdSompi,
      });
      return [entry];
    } catch (error) {
      if (!isLikelyInsufficientFundsError(error)) {
        throw error;
      }
    }
  }

  const prioritized = dedupeUtxos([
    ...normalizeUtxoList(availablePendingUtxos),
    ...normalizeUtxoList(matureUtxos),
  ]).sort(compareContextualUtxos);

  if (prioritized.length === 0) {
    throw new Error("Insufficient funds after fee");
  }

  const selected = [];
  for (const entry of prioritized) {
    selected.push(entry);
    try {
      previewRawDirectedSpend({
        entries: selected,
        amountSompi,
        payloadBytes,
        networkId,
        destinationScriptPublicKey,
        changeScriptPublicKey,
        feeRateSompiPerGram,
        dustThresholdSompi,
      });
      return [...selected];
    } catch (error) {
      if (!isLikelyInsufficientFundsError(error)) {
        throw error;
      }
    }
  }

  throw new Error("Insufficient funds after fee");
}

export function selectCompactionRawEntries({
  matureUtxos = [],
  pendingUtxos = [],
  networkId,
  scriptPublicKey,
  maxInputs = DEFAULT_COMPACTION_MAX_INPUTS,
  minOutputAmount = DUST_THRESHOLD_SOMPI,
  feeRateSompiPerGram = DEFAULT_LOW_FEE_RATE_SOMPI_PER_GRAM,
  dustThresholdSompi = DUST_THRESHOLD_SOMPI,
}) {
  const confirmed = sortByAmountDescending(dedupeUtxos(matureUtxos));
  const pending = sortByAmountDescending(dedupeUtxos(pendingUtxos));
  const candidates = [...confirmed, ...pending].slice(0, Math.max(2, maxInputs));

  if (candidates.length < 2) {
    throw new Error("Not enough spendable UTXOs for compaction");
  }

  let bestSelection = null;
  const selected = [];
  for (const entry of candidates) {
    selected.push(entry);
    if (selected.length < 2) {
      continue;
    }

    try {
      const preview = previewRawSelfSpend({
        entries: selected,
        payloadBytes: new Uint8Array(),
        networkId,
        scriptPublicKey,
        feeRateSompiPerGram,
        dustThresholdSompi,
      });
      if (preview.outputAmount < minOutputAmount) {
        continue;
      }
      if (
        !bestSelection ||
        selected.length > bestSelection.entries.length ||
        (selected.length === bestSelection.entries.length &&
          preview.outputAmount > bestSelection.preview.outputAmount)
      ) {
        bestSelection = {
          entries: [...selected],
          preview,
        };
      }
    } catch (error) {
      if (!isLikelyInsufficientFundsError(error)) {
        throw error;
      }
    }
  }

  if (!bestSelection) {
    throw new Error("Insufficient spendable funds for compaction");
  }

  return bestSelection.entries;
}

function createGeneratorEntries(entries) {
  const normalized = normalizeUtxoList(entries);
  if (normalized.length === 0) {
    return new UtxoEntries([]);
  }
  return new UtxoEntries(normalized);
}

function normalizeAddressValue(value) {
  if (!value) {
    return null;
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value.toString === "function") {
    return value.toString();
  }
  return null;
}

function isLikelyInsufficientFundsError(error) {
  const text = String(error?.message || error || "").toLowerCase();
  return (
    text.includes("insufficient funds") ||
    text.includes("not enough funds") ||
    text.includes("not enough mature") ||
    text.includes("no transaction was produced") ||
    text.includes("not enough balance") ||
    text.includes("storage mass") ||
    text.includes("larger than max allowed size")
  );
}

export function deriveWalletIdentity(seedPhrase, network) {
  const mnemonic = new Mnemonic(String(seedPhrase || "").trim());
  const seed = mnemonic.toSeed("");
  const xprv = new XPrv(seed);
  const privateKeyGenerator = new PrivateKeyGenerator(xprv, false, BigInt(0));
  const privateKey = privateKeyGenerator.receiveKey(0);
  const publicKey = privateKey.toPublicKey();
  const networkId = new NetworkId(network);
  const address = publicKey.toAddress(networkId).toString();
  const scriptPublicKey = payToAddressScript(new Address(address));

  return {
    address,
    privateKey,
    privateKeyHex: privateKey.toString(),
    publicKeyHex: publicKey.toString(),
    scriptPublicKey,
    network,
    networkId,
  };
}

export class KaspaWalletClient {
  constructor({
    seedPhrase,
    nodeUrl,
    network,
    feePolicy = DEFAULT_FEE_POLICY,
    feeEstimateTtlMs = DEFAULT_FEE_ESTIMATE_TTL_MS,
    nowFn = () => Date.now(),
    visibilityRetries = DEFAULT_SEND_VISIBILITY_RETRIES,
    visibilityDelayMs = DEFAULT_SEND_VISIBILITY_DELAY_MS,
    maxConfirmedPlans = DEFAULT_MAX_CONFIRMED_INPUT_PLANS,
    compactionInputThreshold = DEFAULT_COMPACTION_INPUT_THRESHOLD,
    compactionMaxInputs = DEFAULT_COMPACTION_MAX_INPUTS,
    compactionCooldownMs = DEFAULT_COMPACTION_COOLDOWN_MS,
  }) {
    this.seedPhrase = seedPhrase;
    this.nodeUrl = nodeUrl;
    this.network = network || "mainnet";
    this.feePolicy = normalizeFeePolicy(feePolicy);
    this.feeEstimateTtlMs = Math.max(1000, toNumber(feeEstimateTtlMs, DEFAULT_FEE_ESTIMATE_TTL_MS));
    this.nowFn = nowFn;
    this.visibilityRetries = visibilityRetries;
    this.visibilityDelayMs = visibilityDelayMs;
    this.maxConfirmedPlans = maxConfirmedPlans;
    this.compactionInputThreshold = compactionInputThreshold;
    this.compactionMaxInputs = compactionMaxInputs;
    this.compactionCooldownMs = compactionCooldownMs;

    this.identity = null;
    this.rpc = null;
    this.utxoProcessor = null;
    this.utxoContext = null;
    this.isConnected = false;
    this.sendState = normalizeSendState(DEFAULT_SEND_STATE);
    this._sendTail = Promise.resolve();
    this._feeEstimateCache = {
      estimate: null,
      fetched_at_ms: 0,
    };
    this._lastResolvedFeeRate = getFallbackFeeRate(this.feePolicy);
  }

  loadSendState(sendState = {}) {
    this.sendState = normalizeSendState(sendState);
  }

  exportSendState() {
    return normalizeSendState(this.sendState);
  }

  async hydrateSendState() {
    await this._hydrateMempoolSendState();
  }

  async init() {
    this.identity = deriveWalletIdentity(this.seedPhrase, this.network);
    this.rpc = new RpcClient({
      url: this.nodeUrl,
      networkId: this.identity.networkId,
      encoding: Encoding.Borsh,
    });

    await this.rpc.connect({
      blockAsyncConnect: true,
      strategy: ConnectStrategy.Fallback,
      url: this.nodeUrl,
      retryInterval: 2000,
      timeoutDuration: 5000,
    });

    await this._rebuildUtxoContext();

    this.isConnected = true;
    return this.getWalletInfo();
  }

  getWalletInfo() {
    if (!this.identity) {
      throw new Error("Wallet client is not initialized");
    }
    return {
      address: this.identity.address,
      publicKeyHex: this.identity.publicKeyHex,
      privateKeyHex: this.identity.privateKeyHex,
      network: this.identity.network,
    };
  }

  getFeePolicy() {
    return this.feePolicy;
  }

  getLastResolvedFeeRate() {
    return this._lastResolvedFeeRate;
  }

  async resolveFeeRate(policy = this.feePolicy) {
    const normalizedPolicy = normalizeFeePolicy(policy, this.feePolicy);
    const fallbackRate = getFallbackFeeRate(normalizedPolicy);
    if (!this.rpc?.getFeeEstimate) {
      this._lastResolvedFeeRate = fallbackRate;
      return fallbackRate;
    }

    try {
      const estimate = await this._getFeeEstimate();
      const resolvedRate = selectFeeRateFromEstimate(estimate, normalizedPolicy);
      if (resolvedRate != null) {
        this._lastResolvedFeeRate = resolvedRate;
        return resolvedRate;
      }
    } catch {
      if (this._feeEstimateCache?.estimate) {
        const cachedRate = selectFeeRateFromEstimate(
          this._feeEstimateCache.estimate,
          normalizedPolicy
        );
        if (cachedRate != null) {
          this._lastResolvedFeeRate = cachedRate;
          return cachedRate;
        }
      }
    }

    this._lastResolvedFeeRate = fallbackRate;
    return fallbackRate;
  }

  async _getFeeEstimate() {
    const nowMs = this.nowFn();
    if (
      this._feeEstimateCache?.estimate &&
      this._feeEstimateCache.fetched_at_ms + this.feeEstimateTtlMs > nowMs
    ) {
      return this._feeEstimateCache.estimate;
    }

    const estimate = await this.rpc.getFeeEstimate();
    this._feeEstimateCache = {
      estimate,
      fetched_at_ms: nowMs,
    };
    return estimate;
  }

  canFitContextualPayload(payloadBytes) {
    if (!this.identity?.scriptPublicKey) {
      throw new Error("Wallet client is not initialized");
    }

    try {
      previewRawSelfSpend({
        entries: [
          {
            address: this.identity.address,
            amount: SYNTHETIC_PREVIEW_INPUT_AMOUNT_SOMPI,
            scriptPublicKey: this.identity.scriptPublicKey,
            blockDaaScore: 1n,
            isCoinbase: false,
            outpoint: {
              transactionId: "0".repeat(64),
              index: 0,
            },
          },
        ],
        payloadBytes,
        networkId: this.identity.networkId,
        scriptPublicKey: this.identity.scriptPublicKey,
      });
      return true;
    } catch {
      return false;
    }
  }

  async sendPayloadTransaction({
    destinationAddress,
    amountSompi,
    payloadBytes,
    priorityFeeSompi = 0n,
    feePolicy = this.feePolicy,
    strategy = "default",
  }) {
    return await this._enqueueSend(async () =>
      this._sendPayloadTransaction({
        destinationAddress,
        amountSompi,
        payloadBytes,
        priorityFeeSompi,
        feePolicy,
        strategy,
      })
    );
  }

  async _enqueueSend(operation) {
    const previous = this._sendTail;
    let release = null;
    this._sendTail = new Promise((resolve) => {
      release = resolve;
    });

    try {
      await previous.catch(() => {});
      return await operation();
    } finally {
      release?.();
    }
  }

  async _sendPayloadTransaction({
    destinationAddress,
    amountSompi,
    payloadBytes,
    priorityFeeSompi = 0n,
    feePolicy = this.feePolicy,
    strategy = "default",
  }) {
    if (!this.identity || !this.utxoContext || !this.rpc) {
      throw new Error("Wallet client is not initialized");
    }

    const receiveAddress = String(this.identity.address);
    const destination = String(destinationAddress || "").trim();
    const isSelfSend = destination === receiveAddress;
    const sendStrategy =
      strategy === "default" ? (isSelfSend ? "contextual" : "direct") : strategy;
    const feeRateSompiPerGram = await this.resolveFeeRate(feePolicy);

    if (sendStrategy === "contextual") {
      return this._sendRawContextualPayloadTransaction({
        payloadBytes,
        feeRateSompiPerGram,
      });
    }

    return this._sendRawDirectPayloadTransaction({
      destination,
      amountSompi,
      payloadBytes,
      feeRateSompiPerGram,
    });
  }

  async _sendRawDirectPayloadTransaction({
    destination,
    amountSompi,
    payloadBytes,
    feeRateSompiPerGram,
  }) {
    let lastError = null;
    for (let attempt = 0; attempt <= this.visibilityRetries; attempt += 1) {
      await this._rebuildUtxoContext();
      const context = this._loadSendContext();
      const availablePendingUtxos = Array.isArray(context.availablePendingUtxos)
        ? context.availablePendingUtxos
        : [];
      const trackedPendingUtxos = Array.isArray(context.trackedPendingUtxos)
        ? context.trackedPendingUtxos
        : [];
      const availableMatureUtxos = Array.isArray(context.availableMatureUtxos)
        ? context.availableMatureUtxos
        : [];

      try {
        const selectedEntries = selectDirectedRawEntries({
          availablePendingUtxos,
          trackedPendingUtxos,
          matureUtxos: availableMatureUtxos,
          amountSompi,
          payloadBytes,
          networkId: this.identity.networkId,
          destinationScriptPublicKey: payToAddressScript(new Address(destination)),
          changeScriptPublicKey: this.identity.scriptPublicKey,
          feeRateSompiPerGram,
        });
        const submission = await this._submitRawDirectedSpend({
          entries: selectedEntries,
          destinationAddress: destination,
          amountSompi,
          payloadBytes,
          feeRateSompiPerGram,
        });
        return this._finalizeSubmittedTransactions([submission]);
      } catch (error) {
        if (!isLikelyInsufficientFundsError(error)) {
          throw error;
        }
        lastError = error;
      }

      if (
        shouldCompactSend({
          matureUtxos: availableMatureUtxos,
          trackedPendingUtxos,
          lastCompactionMs: this.sendState.last_compaction_ms,
          nowMs: this.nowFn(),
          cooldownMs: this.compactionCooldownMs,
          threshold: this.compactionInputThreshold,
        })
      ) {
        await this._compactSpendableUtxosRaw({
          matureUtxos: availableMatureUtxos,
          pendingUtxos: availablePendingUtxos,
          feeRateSompiPerGram,
        });
        continue;
      }

      if (
        this.sendState.pending_outputs.length > 0 &&
        trackedPendingUtxos.length === 0 &&
        attempt < this.visibilityRetries
      ) {
        await delay(this.visibilityDelayMs * (attempt + 1));
        continue;
      }

      break;
    }

    if (lastError) {
      throw lastError;
    }
    throw new Error(
      "No transaction was produced. The Kasia wallet may not have enough mature balance."
    );
  }

  async _sendRawContextualPayloadTransaction({
    payloadBytes,
    feeRateSompiPerGram,
  }) {
    let lastError = null;

    for (let attempt = 0; attempt <= this.visibilityRetries; attempt += 1) {
      await this._rebuildUtxoContext();
      const context = this._loadSendContext();
      const availablePendingUtxos = Array.isArray(context.availablePendingUtxos)
        ? context.availablePendingUtxos
        : [];
      const trackedPendingUtxos = Array.isArray(context.trackedPendingUtxos)
        ? context.trackedPendingUtxos
        : [];
      const availableMatureUtxos = Array.isArray(context.availableMatureUtxos)
        ? context.availableMatureUtxos
        : [];

      if (
        shouldCompactSend({
          matureUtxos: availableMatureUtxos,
          trackedPendingUtxos,
          lastCompactionMs: this.sendState.last_compaction_ms,
          nowMs: this.nowFn(),
          cooldownMs: this.compactionCooldownMs,
          threshold: this.compactionInputThreshold,
        })
      ) {
        try {
          await this._compactSpendableUtxosRaw({
            matureUtxos: availableMatureUtxos,
            pendingUtxos: availablePendingUtxos,
            feeRateSompiPerGram,
          });
          continue;
        } catch (error) {
          if (!isLikelyInsufficientFundsError(error)) {
            throw error;
          }
          lastError = error;
        }
      }

      try {
        const selectedEntries = selectContextualRawEntries({
          availablePendingUtxos,
          trackedPendingUtxos,
          matureUtxos: availableMatureUtxos,
          payloadBytes,
          networkId: this.identity.networkId,
          scriptPublicKey: this.identity.scriptPublicKey,
          feeRateSompiPerGram,
        });
        const submission = await this._submitRawSelfSpend(
          selectedEntries,
          payloadBytes,
          feeRateSompiPerGram
        );
        return this._finalizeSubmittedTransactions([submission]);
      } catch (error) {
        if (!isLikelyInsufficientFundsError(error)) {
          throw error;
        }
        lastError = error;
      }

      if (attempt < this.visibilityRetries) {
        await delay(this.visibilityDelayMs * (attempt + 1));
      }
    }

    if (lastError) {
      throw lastError;
    }
    throw new Error(
      "No transaction was produced. The Kasia wallet may not have enough mature balance."
    );
  }

  _loadSendContext() {
    const matureUtxos = normalizeUtxoList(
      this.utxoContext.getMatureRange(0, this.utxoContext.matureLength)
    ).filter(isSpendableUtxo);
    const pendingUtxos = normalizeUtxoList(
      this.utxoContext.getPending()
    ).filter(isSpendableUtxo);
    const liveUtxos = [...matureUtxos, ...pendingUtxos];

    this.sendState = reconcileSendState(this.sendState, liveUtxos, {
      nowMs: this.nowFn(),
    });

    const reservedKeys = new Set(
      this.sendState.reserved_outpoints.map((entry) => entry.key)
    );
    const trackedPendingKeys = getTrackedPendingKeys(this.sendState);

    const availableMatureUtxos = matureUtxos.filter((entry) => {
      const key = makeOutpointKey(entry);
      return key && !reservedKeys.has(key);
    });
    const livePendingUtxos = pendingUtxos.filter((entry) => {
      const key = makeOutpointKey(entry);
      return key && !reservedKeys.has(key);
    });
    const syntheticPendingUtxos = this.sendState.pending_outputs
      .filter((entry) => entry.key && !reservedKeys.has(entry.key))
      .map((entry) => this._buildSyntheticPendingUtxo(entry))
      .filter(Boolean);
    const availablePendingUtxos = dedupeUtxos([
      ...syntheticPendingUtxos,
      ...livePendingUtxos,
    ]);
    const trackedPendingUtxos = sortByAmountDescending(
      availablePendingUtxos.filter((entry) =>
        trackedPendingKeys.has(makeOutpointKey(entry))
      )
    );

    return {
      availableMatureUtxos,
      availablePendingUtxos,
      trackedPendingUtxos,
    };
  }

  _buildSyntheticPendingUtxo(entry) {
    const transactionId = String(entry?.tx_id || entry?.transaction_id || "").trim();
    const index = toNumber(entry?.index, -1);
    if (!transactionId || index < 0 || !this.identity?.scriptPublicKey) {
      return null;
    }
    return {
      address: this.identity.address,
      amount: toBigInt(entry?.amount, 0n),
      scriptPublicKey: this.identity.scriptPublicKey,
      blockDaaScore: 0n,
      isCoinbase: false,
      outpoint: {
        transactionId,
        index,
      },
    };
  }

  async _submitRawSelfSpend(entries, payloadBytes, feeRateSompiPerGram) {
    const rawTransaction = buildRawSelfSpendTransaction({
      entries,
      payloadBytes,
      networkId: this.identity.networkId,
      scriptPublicKey: this.identity.scriptPublicKey,
      privateKey: this.identity.privateKey,
      feeRateSompiPerGram,
    });
    const txId = await this._submitRawTransaction(
      rawTransaction.transaction,
      rawTransaction.usesPendingInputs
    );
    this._reserveConsumedInputs(entries);
    this._consumePendingOutputs(entries);
    this._trackPendingOutputs(rawTransaction.transaction, txId);
    return {
      txId,
      inputCount: rawTransaction.inputCount,
      usesPendingInputs: rawTransaction.usesPendingInputs,
    };
  }

  async _submitRawDirectedSpend({
    entries,
    destinationAddress,
    amountSompi,
    payloadBytes,
    feeRateSompiPerGram,
  }) {
    const rawTransaction = buildRawDirectedSpendTransaction({
      entries,
      amountSompi,
      payloadBytes,
      networkId: this.identity.networkId,
      destinationScriptPublicKey: payToAddressScript(
        new Address(destinationAddress)
      ),
      changeScriptPublicKey: this.identity.scriptPublicKey,
      privateKey: this.identity.privateKey,
      feeRateSompiPerGram,
    });
    const txId = await this._submitRawTransaction(
      rawTransaction.transaction,
      rawTransaction.usesPendingInputs
    );
    this._reserveConsumedInputs(entries);
    this._consumePendingOutputs(entries);
    this._trackPendingOutputs(rawTransaction.transaction, txId);
    return {
      txId,
      inputCount: rawTransaction.inputCount,
      usesPendingInputs: rawTransaction.usesPendingInputs,
    };
  }

  async _submitRawTransaction(transaction, usesPendingInputs) {
    if (usesPendingInputs) {
      const response = await this.rpc.submitTransaction({
        transaction,
        allowOrphan: true,
      });
      return this._extractTransactionIdFromResponse(response, transaction);
    }

    const response = await this.rpc.submitTransaction({
      transaction,
      allowOrphan: false,
    });
    return this._extractTransactionIdFromResponse(response, transaction);
  }

  _extractTransactionIdFromResponse(response, transaction) {
    if (typeof response === "string" && response.trim()) {
      return response;
    }
    if (response && typeof response === "object") {
      if (typeof response.txId === "string" && response.txId.trim()) {
        return response.txId;
      }
      if (
        typeof response.transactionId === "string" &&
        response.transactionId.trim()
      ) {
        return response.transactionId;
      }
      if (typeof response.id === "string" && response.id.trim()) {
        return response.id;
      }
    }
    const txId = normalizeAddressValue(transaction?.id);
    if (txId) {
      return txId;
    }
    throw new Error("Submitted transaction did not return a transaction id");
  }

  async _compactSpendableUtxosRaw({
    matureUtxos,
    pendingUtxos,
    feeRateSompiPerGram,
  }) {
    const selectedEntries = selectCompactionRawEntries({
      matureUtxos,
      pendingUtxos,
      networkId: this.identity.networkId,
      scriptPublicKey: this.identity.scriptPublicKey,
      maxInputs: this.compactionMaxInputs,
      feeRateSompiPerGram,
    });
    await this._submitRawSelfSpend(
      selectedEntries,
      new Uint8Array(),
      feeRateSompiPerGram
    );
    this.sendState.last_compaction_ms = this.nowFn();
    await delay(this.visibilityDelayMs);
  }

  async _tryPlans(plans, txOptions) {
    let lastError = null;
    for (const plan of plans) {
      if (
        !plan.useFullContext &&
        (!Array.isArray(plan.entries) || plan.entries.length === 0)
      ) {
        continue;
      }
      try {
        const transactions = await this._submitPlan(plan, txOptions);
        return {
          success: true,
          transactions,
        };
      } catch (error) {
        if (!isLikelyInsufficientFundsError(error)) {
          throw error;
        }
        lastError = error;
      }
    }
    return {
      success: false,
      error: lastError,
    };
  }

  async _submitPlan(
    plan,
    { destination, receiveAddress, amountSompi, payloadBytes, priorityFeeSompi }
  ) {
    const receiveAddressObject = new Address(receiveAddress);
    const outputs =
      destination === receiveAddress
        ? []
        : [new PaymentOutput(new Address(destination), toBigInt(amountSompi, 0n))];
    const generator = new Generator({
      changeAddress: receiveAddressObject,
      entries: plan.useFullContext
        ? this.utxoContext
        : createGeneratorEntries(plan.entries),
      outputs,
      payload: payloadBytes,
      networkId: this.identity.networkId,
      priorityFee: toBigInt(priorityFeeSompi, 0n),
    });

    const submissions = [];
    let pendingTransaction;
    while ((pendingTransaction = await generator.next())) {
      const selectedEntries = normalizeUtxoList(
        pendingTransaction.getUtxoEntries()
      );
      const usesPendingInputs =
        plan.usesPendingInputs || selectedEntries.some(isPendingUtxo);
      const txId = await this._submitPendingTransaction(
        pendingTransaction,
        usesPendingInputs
      );
      this._reserveConsumedInputs(selectedEntries);
      this._consumePendingOutputs(selectedEntries);
      this._trackPendingOutputs(pendingTransaction.transaction, txId);
      submissions.push({
        txId,
        inputCount: selectedEntries.length,
        usesPendingInputs,
      });
    }

    if (submissions.length === 0) {
      throw new Error(
        "No transaction was produced. The Kasia wallet may not have enough mature balance."
      );
    }

    return submissions;
  }

  async _submitPendingTransaction(pendingTransaction, usesPendingInputs) {
    pendingTransaction.sign([this.identity.privateKey], false);

    if (usesPendingInputs) {
      try {
        const response = await this.rpc.submitTransaction({
          transaction: pendingTransaction.transaction,
          allowOrphan: true,
        });
        return this._extractTxId(response, pendingTransaction);
      } catch {
        // Fall back to the runtime helper if the request shape changes between SDK releases.
      }
    }

    const response = await pendingTransaction.submit(this.rpc);
    return this._extractTxId(response, pendingTransaction);
  }

  _extractTxId(response, pendingTransaction) {
    if (typeof response === "string" && response.trim()) {
      return response;
    }
    if (response && typeof response === "object") {
      if (typeof response.txId === "string" && response.txId.trim()) {
        return response.txId;
      }
      if (typeof response.transactionId === "string" && response.transactionId.trim()) {
        return response.transactionId;
      }
      if (typeof response.id === "string" && response.id.trim()) {
        return response.id;
      }
    }
    const txId =
      normalizeAddressValue(pendingTransaction?.transaction?.id) ||
      normalizeAddressValue(pendingTransaction?.id);
    if (txId) {
      return txId;
    }
    throw new Error("Submitted transaction did not return a transaction id");
  }

  _reserveConsumedInputs(entries) {
    const nowMs = this.nowFn();
    const byKey = new Map(
      this.sendState.reserved_outpoints.map((entry) => [entry.key, entry])
    );
    for (const entry of normalizeUtxoList(entries)) {
      const key = makeOutpointKey(entry);
      if (!key) {
        continue;
      }
      byKey.set(key, {
        key,
        reserved_at_ms: nowMs,
      });
    }
    this.sendState.reserved_outpoints = [...byKey.values()];
  }

  _consumePendingOutputs(entries) {
    const consumedKeys = new Set(
      normalizeUtxoList(entries)
        .map((entry) => makeOutpointKey(entry))
        .filter(Boolean)
    );
    this.sendState.pending_outputs = this.sendState.pending_outputs.filter(
      (entry) => !consumedKeys.has(entry.key)
    );
  }

  _trackPendingOutputs(transaction, txId) {
    if (!transaction || !Array.isArray(transaction.outputs)) {
      return;
    }

    const nowMs = this.nowFn();
    const byKey = new Map(
      this.sendState.pending_outputs.map((entry) => [entry.key, entry])
    );

    transaction.outputs.forEach((output, index) => {
      const address = this._tryOutputAddress(output);
      if (address !== this.identity.address) {
        return;
      }
      const key = makeOutpointKey(txId, index);
      if (!key) {
        return;
      }
      byKey.set(key, {
        key,
        tx_id: txId,
        index,
        amount: String(toBigInt(output?.value, 0n)),
        created_ms: nowMs,
        observed_in_mempool: false,
      });
    });

    this.sendState.pending_outputs = [...byKey.values()];
  }

  _tryOutputAddress(output) {
    try {
      return normalizeAddressValue(
        addressFromScriptPublicKey(output?.scriptPublicKey, this.identity.networkId)
      );
    } catch {
      return null;
    }
  }

  async _compactMatureUtxos(availableMatureUtxos, receiveAddress) {
    if (availableMatureUtxos.length < 2) {
      return;
    }

    await this._submitPlan(
      {
        name: "compaction",
        useFullContext: true,
        usesPendingInputs: false,
      },
      {
        destination: receiveAddress,
        receiveAddress,
        amountSompi: 0n,
        payloadBytes: new Uint8Array(),
        priorityFeeSompi: 0n,
      }
    );
    this.sendState.last_compaction_ms = this.nowFn();
    await delay(this.visibilityDelayMs);
  }

  _finalizeSubmittedTransactions(transactions) {
    const last = transactions[transactions.length - 1];
    return {
      txId: last.txId,
      transactionCount: transactions.length,
      inputCount: transactions.reduce(
        (total, entry) => total + toNumber(entry.inputCount, 0),
        0
      ),
      usedPendingInput: transactions.some((entry) => entry.usesPendingInputs),
      sendState: this.exportSendState(),
    };
  }

  async close() {
    this.isConnected = false;

    try {
      await this.utxoContext?.clear?.();
    } catch {}
    try {
      await this.utxoProcessor?.stop?.();
    } catch {}
    try {
      await this.rpc?.disconnect?.();
    } catch {}
  }

  async _getOnChainBalance() {
    try {
      const balance = await this.rpc?.getBalanceByAddress?.({
        address: this.identity?.address,
      });
      return toBigInt(balance?.balance, 0n);
    } catch {
      return 0n;
    }
  }

  async _rebuildUtxoContext() {
    try {
      await this.utxoContext?.clear?.();
    } catch {}
    try {
      await this.utxoProcessor?.stop?.();
    } catch {}

    this.utxoProcessor = new UtxoProcessor({
      rpc: this.rpc,
      networkId: this.identity.networkId,
    });
    await this.utxoProcessor.start();

    this.utxoContext = new UtxoContext({
      processor: this.utxoProcessor,
    });
    await this.utxoContext.trackAddresses([this.identity.address]);
    await this._hydrateMempoolSendState();

    const chainBalance = await this._getOnChainBalance();
    if (chainBalance <= 0n) {
      return;
    }

    for (let attempt = 0; attempt < 10; attempt += 1) {
      if (
        this.utxoContext.matureLength > 0 ||
        normalizeUtxoList(this.utxoContext.getPending()).length > 0
      ) {
        return;
      }
      await delay(500);
    }
  }

  async _hydrateMempoolSendState() {
    try {
      const mempool = await this.rpc?.getMempoolEntriesByAddresses?.({
        addresses: [this.identity.address],
        includeOrphanPool: true,
        filterTransactionPool: false,
      });
      const addressEntry = mempoolEntriesForAddress(mempool, this.identity.address);
      const nowMs = this.nowFn();
      const retentionMs = Math.max(
        DEFAULT_LOCAL_PENDING_RETENTION_MS,
        this.visibilityRetries * this.visibilityDelayMs * 2
      );
      const reservedByKey = new Map(
        this.sendState.reserved_outpoints
          .filter((entry) => entry.reserved_at_ms + retentionMs > nowMs)
          .map((entry) => [entry.key, entry])
      );
      const pendingByKey = new Map(
        this.sendState.pending_outputs
          .filter((entry) => entry.created_ms + retentionMs > nowMs)
          .map((entry) => [entry.key, entry])
      );

      for (const mempoolEntry of addressEntry?.sending || []) {
        const transaction = mempoolEntry?.transaction;
        for (const input of transaction?.inputs || []) {
          const key = makeOutpointKey(
            input?.previousOutpoint?.transactionId,
            input?.previousOutpoint?.index
          );
          if (!key) {
            continue;
          }
          reservedByKey.set(key, {
            key,
            reserved_at_ms: nowMs,
          });
        }

        for (const pendingOutput of pendingOutputsFromMempoolEntry(
          mempoolEntry,
          this.identity.address,
          this.identity.networkId,
          nowMs
        )) {
          if (reservedByKey.has(pendingOutput.key)) {
            continue;
          }
          pendingByKey.set(pendingOutput.key, pendingOutput);
        }
      }

      for (const reservedKey of reservedByKey.keys()) {
        pendingByKey.delete(reservedKey);
      }

      this.sendState = normalizeSendState({
        ...this.sendState,
        reserved_outpoints: [...reservedByKey.values()],
        pending_outputs: [...pendingByKey.values()],
      });
    } catch {
      // Mempool hydration is a best-effort recovery path for restarts.
    }
  }
}
