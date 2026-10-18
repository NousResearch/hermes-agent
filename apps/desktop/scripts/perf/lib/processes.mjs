// Per-process CPU sampling for the perf harness, via CDP.
//
// The idle-burn story (#127647) lives in processes, not in the renderer: the
// GPU helper burning ~1.5 cores under the SwiftShader pin shows up as near-zero
// renderer cost while a single composited animation renders unthrottled. No
// renderer-side counter can see that, so the scenario needs per-process
// numbers: CDP `SystemInfo.getProcessInfo` (browser-scoped domain) returns
// every Chromium child with its cumulative `cpuTime` in seconds — the metric
// the tracker's rules ask for (CPU-time deltas over fixed windows, not %CPU).
//
// Two transports, tried in order:
//   1. the page-scoped CDP connection the harness already holds — Electron
//      serves the SystemInfo domain there too (fastest, needs nothing extra);
//   2. a fresh WebSocket to the browser-level target from `/json/version`
//      (what the DevTools spec formally scopes the domain to).
// A sample is `{ pid: number, label: string, cpu: number }[]`; cpu is cumulative
// seconds since process start. Callers take two samples and diff them.

/** How long to wait for a single CDP round-trip before giving up on a transport. */
const CDP_TIMEOUT_MS = 5000

export function labelForProcess(row) {
  const type = String(row.type ?? '').toLowerCase()
  const name = String(row.name ?? '')

  if (type === 'browser') return 'main'
  if (type === 'gpu' || type.includes('gpu')) return 'gpu'
  if (type === 'renderer') return 'renderer'
  if (type.includes('zygote')) return 'zygote'
  if (type.includes('utility') || type.includes('service')) return name || 'utility'
  return name || type
}

function withTimeout(promise, ms, what) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`${what} timed out after ${ms}ms`)), ms))
  ])
}

/** One CDP command over a raw WebSocket, resolving `{ result, error }`. */
function sendCdp(ws, method, params, idCounter) {
  return new Promise((resolveFn, reject) => {
    const id = ++idCounter.n
    const onMessage = event => {
      const data = JSON.parse(typeof event.data === 'string' ? event.data : String(event.data))
      if (data.id === id) {
        ws.removeEventListener('message', onMessage)
        resolveFn(data)
      }
    }

    ws.addEventListener('message', onMessage)
    ws.send(JSON.stringify({ id, method, params: params ?? {} }))
    // Id-mismatched replies leave the listener; the timeout below is the guard.
  })
}

async function sampleViaBrowserTarget(port) {
  const version = await withTimeout(
    (await fetch(`http://127.0.0.1:${port}/json/version`)).json(),
    CDP_TIMEOUT_MS,
    '/json/version'
  )
  const ws = new WebSocket(version.webSocketDebuggerUrl)

  try {
    await withTimeout(
      new Promise((resolveFn, reject) => { ws.addEventListener('open', resolveFn, { once: true }); ws.addEventListener('error', reject, { once: true }) }),
      CDP_TIMEOUT_MS,
      'browser target open'
    )
    const res = await withTimeout(sendCdp(ws, 'SystemInfo.getProcessInfo', {}, { n: 0 }), CDP_TIMEOUT_MS, 'SystemInfo.getProcessInfo')

    if (!res.result?.processInfo) {
      throw new Error(res.error?.message || 'no processInfo')
    }

    return res.result.processInfo
  } finally {
    try { ws.close() } catch { /* already gone */ }
  }
}

async function sampleViaPageTarget(cdp) {
  if (typeof cdp?.send !== 'function') {
    throw new Error('not a CDP connection')
  }

  const res = await cdp.send('SystemInfo.getProcessInfo')

  if (!res?.processInfo) {
    throw new Error('no processInfo')
  }

  return res.processInfo
}

/**
 * Sample per-process CPU counters for every Chromium child of the instance.
 *
 * @param {import('./cdp.mjs').CDP} cdp the harness's page-scoped CDP connection
 * @param {{port?: number}} [options] `port` is the instance's CDP HTTP port,
 *   used only for the browser-target fallback
 * @returns {Promise<Array<{pid: number, label: string, cpu: number}>>}
 *   cumulative CPU seconds per process; empty when no transport served the
 *   domain (the scenario reports that as "unavailable" instead of guessing)
 */
export async function sampleProcessCpu(cdp, { port } = {}) {
  let rows = null

  try {
    rows = await sampleViaPageTarget(cdp)
  } catch {
    if (port) {
      rows = await sampleViaBrowserTarget(port)
    }
  }

  if (!rows) {
    return []
  }

  return rows
    .filter(row => row.type !== 'other')
    .map(row => ({
      pid: Number(row.processId ?? row.pid ?? 0) || null,
      label: labelForProcess(row),
      cpu: Number(row.cpuTime ?? 0)
    }))
}

/**
 * Diff two samples into CPU-second deltas per label. Rows carry no pid in
 * current Chromium (`processInfo.processId` is absent), so matching is by
 * label sums — which also nets per label across a GPU-child respawn.
 */
export function diffProcessCpu(before, after) {
  const sumBy = rows => {
    const out = new Map()

    for (const proc of rows) {
      out.set(proc.label, (out.get(proc.label) ?? 0) + proc.cpu)
    }

    return out
  }

  const beforeByLabel = sumBy(before)
  const afterByLabel = sumBy(after)
  const out = new Map()

  for (const [label, cpu] of afterByLabel) {
    out.set(label, Math.round(Math.max(0, cpu - (beforeByLabel.get(label) ?? 0)) * 100) / 100)
  }

  return Object.fromEntries([...out.entries()].sort(([a], [b]) => a.localeCompare(b)))
}
