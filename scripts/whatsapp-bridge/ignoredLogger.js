const DEFAULT_IGNORED_LOG_INTERVAL_MS = 60_000;

export function createIgnoredMessageLogger({
  debugEnabled = false,
  log = console.log,
  now = Date.now,
  intervalMs = DEFAULT_IGNORED_LOG_INTERVAL_MS,
} = {}) {
  const lastLoggedAtByReason = new Map();

  return function logIgnoredMessage(reason) {
    if (!debugEnabled || !reason) {
      return;
    }

    const currentTime = now();
    const lastLoggedAt = lastLoggedAtByReason.get(reason);
    if (lastLoggedAt !== undefined && currentTime - lastLoggedAt < intervalMs) {
      return;
    }

    lastLoggedAtByReason.set(reason, currentTime);
    try {
      log(JSON.stringify({ event: 'ignored', reason }));
    } catch {}
  };
}
