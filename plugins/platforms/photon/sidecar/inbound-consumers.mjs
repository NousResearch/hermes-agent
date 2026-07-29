function waitForDrain(state) {
  const { response } = state;
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      response.off("drain", onDrain);
      response.off("close", onClose);
      response.off("error", onError);
      state.cancelDrain = null;
    };
    const onDrain = () => {
      cleanup();
      resolve();
    };
    const onClose = () => {
      cleanup();
      reject(new Error("consumer closed before drain"));
    };
    const onError = (error) => {
      cleanup();
      reject(error);
    };

    response.once("drain", onDrain);
    response.once("close", onClose);
    response.once("error", onError);
    state.cancelDrain = () => {
      cleanup();
      reject(new Error("consumer removed before drain"));
    };
  });
}

export function createConsumerHub({
  maxQueueLines = 256,
  maxQueueBytes = 32 * 1024 * 1024,
  maxOrphanLines = 256,
  maxOrphanBytes = maxQueueBytes,
} = {}) {
  const consumers = new Map();
  const orphanQueue = [];
  let orphanBytes = 0;

  function remove(response, { destroy = false } = {}) {
    const state = consumers.get(response);
    if (!state) return;
    state.closed = true;
    state.queue.length = 0;
    state.queueBytes = 0;
    consumers.delete(response);
    if (state.cancelDrain) state.cancelDrain();
    if (destroy && typeof response.destroy === "function") {
      response.destroy();
    }
  }

  async function flush(state) {
    if (state.flushing || state.closed) return;
    state.flushing = true;
    try {
      while (!state.closed && state.queue.length > 0) {
        const entry = state.queue[0];
        const flushed = state.response.write(entry.line + "\n");
        if (!flushed) await waitForDrain(state);
        state.queue.shift();
        state.queueBytes -= entry.bytes;
      }
    } catch {
      remove(state.response);
    } finally {
      state.flushing = false;
      if (!state.closed && state.queue.length > 0) void flush(state);
    }
  }

  function enqueue(state, line, bytes = Buffer.byteLength(line) + 1) {
    if (state.closed) return false;
    if (
      state.queue.length >= maxQueueLines ||
      (state.queue.length > 0 && state.queueBytes + bytes > maxQueueBytes)
    ) {
      remove(state.response, { destroy: true });
      return false;
    }
    state.queue.push({ line, bytes });
    state.queueBytes += bytes;
    void flush(state);
    return true;
  }

  function add(response) {
    if (consumers.has(response)) return;
    const state = {
      response,
      queue: [],
      queueBytes: 0,
      flushing: false,
      closed: false,
      cancelDrain: null,
    };
    consumers.set(response, state);

    // Preserve a small bounded window across a momentary all-consumers-down
    // reconnect without ever stalling the upstream Spectrum iterator.
    const pending = orphanQueue.splice(0);
    orphanBytes = 0;
    for (const entry of pending) enqueue(state, entry.line, entry.bytes);
  }

  async function deliver(line) {
    const bytes = Buffer.byteLength(line) + 1;
    if (consumers.size === 0) {
      orphanQueue.push({ line, bytes });
      orphanBytes += bytes;
      while (
        orphanQueue.length > maxOrphanLines ||
        (orphanQueue.length > 1 && orphanBytes > maxOrphanBytes)
      ) {
        orphanBytes -= orphanQueue.shift().bytes;
      }
      return 0;
    }

    let accepted = 0;
    for (const state of [...consumers.values()]) {
      if (enqueue(state, line, bytes)) accepted += 1;
    }
    return accepted;
  }

  return { add, remove, deliver };
}
