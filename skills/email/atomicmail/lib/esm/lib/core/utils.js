// Small async helpers (delay, exponential backoff retry).
import { ONE_SEC_MS } from "./consts.js";
export function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
const defaultCfg = {
    maxTimeoutMs: ONE_SEC_MS * 32,
    startTimeoutMs: ONE_SEC_MS,
    backoffMul: 2,
};
/** Retries `fn` on throw with exponential backoff until `maxTimeoutMs` is exceeded. */
export async function retry(fn, config) {
    const cfg = { ...defaultCfg, ...config };
    let curTimeoutMs = cfg.startTimeoutMs;
    while (true) {
        try {
            const res = await fn();
            return res;
        }
        catch (e) {
            if (cfg.onBeforeRetry)
                await cfg.onBeforeRetry(e);
            if (curTimeoutMs > cfg.maxTimeoutMs)
                throw e;
            await delay(curTimeoutMs);
            curTimeoutMs = Math.floor(curTimeoutMs * cfg.backoffMul);
        }
    }
}
