// Retry an interrupted inbound download in full, before the bridge publishes
// the message or saves a file. Baileys retains ownership of decryption and
// expired-link recovery. Do not retry permanent HTTP failures.
import { setTimeout as sleep } from 'node:timers/promises';

export function isTransientDownloadError(error) {
  const seen = new Set();
  for (let current = error; current && !seen.has(current); current = current.cause) {
    seen.add(current);
    const status = current.status ?? current.statusCode ?? current.output?.statusCode;
    if (typeof status === 'number') {
      return status === 408 || status === 429 || (status >= 500 && status <= 599);
    }
    if (/^(ECONNRESET|ECONNREFUSED|ETIMEDOUT|EPIPE|EAI_AGAIN|UND_ERR_SOCKET|UND_ERR_CONNECT_TIMEOUT|UND_ERR_HEADERS_TIMEOUT|UND_ERR_BODY_TIMEOUT|UND_ERR_RES_CONTENT_LENGTH_MISMATCH)$/.test(current.code || '')) return true;
    if (/^(terminated|fetch failed|premature close)$/i.test(current.message || '')) return true;
  }
  return false;
}

export async function downloadWithRetry(download, {
  delays = [1000, 3000],
  wait = sleep,
  log = console.warn,
} = {}) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await download();
    } catch (error) {
      if (attempt >= delays.length || !isTransientDownloadError(error)) throw error;
      // Log no media URL, message body, sender ID, or credentials.
      log(`[bridge] interrupted inbound media download; retry ${attempt + 1}/${delays.length} in ${delays[attempt]}ms`);
      await wait(delays[attempt]);
    }
  }
}
