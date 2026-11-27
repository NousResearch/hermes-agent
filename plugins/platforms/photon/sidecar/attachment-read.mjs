const DEFAULT_ATTEMPTS = 3;
const DEFAULT_RETRY_MS = 300;

export async function readBinaryContentWithRetry(
  content,
  {
    label = "attachment",
    attempts = DEFAULT_ATTEMPTS,
    retryMs = DEFAULT_RETRY_MS,
    sleep = (delayMs) => new Promise((resolve) => setTimeout(resolve, delayMs)),
    log = console.error,
  } = {}
) {
  const maxAttempts = Math.max(1, Math.trunc(attempts));
  let lastError;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      return await content.read();
    } catch (error) {
      lastError = error;
      if (attempt >= maxAttempts) break;

      const delayMs = retryMs * attempt;
      log(
        `photon-sidecar: failed to read ${label} bytes ` +
          `(attempt ${attempt}/${maxAttempts}); retrying in ${delayMs}ms: ` +
          (error && error.message ? error.message : String(error))
      );
      if (delayMs > 0) await sleep(delayMs);
    }
  }

  throw lastError;
}
