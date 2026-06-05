// A reconnect step (the getConnection IPC, the WS-URL re-mint) must not hang
// forever. After a no-network suspend either await can stall indefinitely,
// which latches a `reconnecting` flag true and silently blocks every future
// backoff tick — the symptom users hit as a permanently frozen composer/sidebar
// after wake. Bound them so a stalled attempt rejects and the backoff loop
// resumes. gateway.connect() already enforces its own connect timeout.
export const RECONNECT_STEP_TIMEOUT_MS = 20_000

export function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)

    promise.then(
      value => {
        clearTimeout(timer)
        resolve(value)
      },
      err => {
        clearTimeout(timer)
        reject(err)
      }
    )
  })
}
