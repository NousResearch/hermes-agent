/**
 * Fixed-window scheduler for receive-only WhatsApp inbox sweeps.
 *
 * A capture attempt starts on each successful connection. The next connection
 * attempt is anchored to that open event, while the WhatsApp companion is
 * always closed after the fixed capture window.
 */
export function createInboxSweepController({
  reconnectIntervalMs,
  windowMs = 3_000,
  setTimeoutFn = setTimeout,
  clearTimeoutFn = clearTimeout,
  closeSocket,
  reconnect,
}) {
  if (!Number.isSafeInteger(reconnectIntervalMs) || reconnectIntervalMs <= windowMs) {
    throw new TypeError('reconnectIntervalMs must exceed windowMs');
  }
  if (!Number.isSafeInteger(windowMs) || windowMs <= 0) {
    throw new TypeError('windowMs must be a positive safe integer');
  }
  if (typeof closeSocket !== 'function' || typeof reconnect !== 'function') {
    throw new TypeError('closeSocket and reconnect must be functions');
  }

  let closeTimer = null;
  let reconnectTimer = null;

  function clear(timer) {
    if (timer !== null) clearTimeoutFn(timer);
    return null;
  }

  return {
    connected() {
      closeTimer = clear(closeTimer);
      reconnectTimer = clear(reconnectTimer);
      closeTimer = setTimeoutFn(() => {
        closeTimer = null;
        closeSocket();
      }, windowMs);
      reconnectTimer = setTimeoutFn(() => {
        reconnectTimer = null;
        reconnect();
      }, reconnectIntervalMs);
    },

    // Receipt arrival intentionally cannot extend the capture window.
    receivedInbound() {},

    closed({ intentional, reason } = {}) {
      closeTimer = clear(closeTimer);
      if (intentional) return;
      reconnectTimer = clear(reconnectTimer);
      reconnectTimer = setTimeoutFn(() => {
        reconnectTimer = null;
        reconnect();
      }, reason === 515 ? 1_000 : 3_000);
    },

    stop() {
      closeTimer = clear(closeTimer);
      reconnectTimer = clear(reconnectTimer);
    },
  };
}

/** Hold bounded sweep receipts until the socket has actually closed. */
export function createInboxReceiptBuffer({ deliver, maxEntries = 100 }) {
  if (typeof deliver !== 'function') throw new TypeError('deliver must be a function');
  if (!Number.isSafeInteger(maxEntries) || maxEntries < 1) {
    throw new TypeError('maxEntries must be a positive safe integer');
  }
  const pending = [];
  return {
    capture(receipt) {
      if (pending.length >= maxEntries) return false;
      pending.push(receipt);
      return true;
    },
    release() {
      const receipts = pending.splice(0);
      for (const receipt of receipts) deliver(receipt);
    },
    get size() { return pending.length; },
  };
}
