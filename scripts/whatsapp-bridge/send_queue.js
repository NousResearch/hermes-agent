// Send queue: serialise all sock.sendMessage() calls across concurrent HTTP
// handlers so a single Baileys socket never has overlapping sends. Overlapping
// sends are the root cause of cross-chat contamination (#33360): the WhatsApp
// protocol-level routing can misdeliver when two sendMessage() Promises race on
// the same socket. Each send also has a deadline (bridge.js explains the default).
export function createSendQueue({ getSocket, defaultTimeoutMs }) {
  let tail = Promise.resolve();

  function enqueueSend(fn) {
    const task = tail.then(() => fn(), () => fn());
    tail = task.catch(() => {});
    return task;
  }

  function deadline(ms) {
    let timer;
    let expired = false;
    const promise = new Promise((_, reject) => {
      timer = setTimeout(() => {
        expired = true;
        reject(new Error(`sendMessage timed out after ${ms / 1000}s`));
      }, ms);
    });
    // A send can expire while it still waits in the queue, before a race is
    // attached; mark the rejection handled so it cannot crash the process. The
    // caller still sees it through the race.
    promise.catch(() => {});
    return { promise, expired: () => expired, clear: () => clearTimeout(timer) };
  }

  // `skipIfExpired` is for best-effort sends (reactions) with a short deadline:
  // a send whose deadline passed while it waited in the queue is dropped instead
  // of started late. One that started keeps the queue until the socket call
  // settles (bounded by the default timeout like every other send), so the
  // caller's short deadline never lets the next send overlap it on the socket.
  function sendWithTimeout(chatId, payload, options = {}, timeoutMs = defaultTimeoutMs, { skipIfExpired = false } = {}) {
    const callerDeadline = deadline(timeoutMs);
    if (!skipIfExpired) {
      return enqueueSend(() => Promise.race([getSocket().sendMessage(chatId, payload, options), callerDeadline.promise])
        .finally(callerDeadline.clear));
    }
    const queued = enqueueSend(() => {
      if (callerDeadline.expired()) return undefined;
      const ceiling = deadline(defaultTimeoutMs);
      return Promise.race([getSocket().sendMessage(chatId, payload, options), ceiling.promise])
        .finally(ceiling.clear);
    });
    queued.catch(() => {});
    return Promise.race([queued, callerDeadline.promise]).finally(callerDeadline.clear);
  }

  return { enqueueSend, sendWithTimeout };
}
