// The one Chrome DevTools Protocol client for desktop scripts.
//
// Before this, every measure-*/profile-* script shipped its own copy-pasted
// `CDP` class (four subtly different implementations), its own `/json` vs
// `/json/list` target discovery, and its own Profiler ranking. Scenarios now
// import from here so there is a single place to fix a protocol bug.

const DEFAULT_PORT = 9222

// Stable DOM hooks the renderer exposes. Centralised so a component refactor
// updates one constant instead of a dozen scattered querySelector strings.
export const SELECTORS = {
  composer: '[data-slot="composer-rich-input"]',
  threadViewport: '[data-slot="aui_thread-viewport"]',
  threadContent: '[data-slot="aui_thread-content"]',
  assistantMessage: '[data-slot="aui_assistant-message-root"]',
  turnPair: '[data-slot="aui_turn-pair"]',
  profileRail: '[data-slot="profile-rail"]',
  rowButton: '[data-slot="row-button"]'
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

async function websocketMessageText(data) {
  if (typeof data === 'string') return data
  if (typeof Blob !== 'undefined' && data instanceof Blob) return data.text()
  if (data instanceof ArrayBuffer) return new TextDecoder().decode(data)
  if (ArrayBuffer.isView(data)) {
    return new TextDecoder().decode(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))
  }

  return data.toString('utf8')
}

/**
 * Poll the CDP HTTP endpoint until a page target is available.
 * @param {object} [opts]
 * @param {number} [opts.port] remote-debugging-port (default 9222).
 * @param {string} [opts.match] substring the target URL must contain (e.g. a dev-server port).
 * @param {number} [opts.timeoutMs] how long to wait for a target.
 */
export async function discoverTarget({ port = DEFAULT_PORT, match, timeoutMs = 30000 } = {}) {
  const deadline = Date.now() + timeoutMs

  for (;;) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()
      const pages = list.filter(t => t.type === 'page' && typeof t.webSocketDebuggerUrl === 'string')
      const target = match
        ? pages.find(t => String(t.url).includes(match))
        : pages.find(t => String(t.url).startsWith('http')) ?? pages[0]

      if (target) {
        return target
      }
    } catch {
      // debug port not up yet — keep polling until the deadline.
    }

    if (Date.now() >= deadline) {
      throw new Error(`no CDP page target on :${port}${match ? ` matching "${match}"` : ''} within ${timeoutMs}ms`)
    }

    await sleep(250)
  }
}

export class CDP {
  constructor(ws) {
    this.ws = ws
    this.id = 0
    this.pending = new Map()
    this.listeners = new Map()
  }

  static async open(url, { timeoutMs = 10_000 } = {}) {
    const ws = new WebSocket(url)

    await new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timeout)
        ws.removeEventListener('open', opened)
        ws.removeEventListener('error', failed)
      }
      const opened = () => {
        cleanup()
        resolve()
      }
      const failed = () => {
        cleanup()
        reject(new Error('CDP websocket failed to open'))
      }
      const timeout = setTimeout(() => {
        cleanup()
        ws.close()
        reject(new Error('CDP websocket open timeout'))
      }, timeoutMs)

      timeout.unref?.()
      ws.addEventListener('open', opened, { once: true })
      ws.addEventListener('error', failed, { once: true })
    })

    const cdp = new CDP(ws)

    ws.addEventListener('message', async ev => {
      const m = JSON.parse(await websocketMessageText(ev.data))

      if (m.id != null && cdp.pending.has(m.id)) {
        const { resolve, reject } = cdp.pending.get(m.id)
        cdp.pending.delete(m.id)

        if (m.error) {
          reject(new Error(m.error.message))
        } else {
          resolve(m.result)
        }
      } else if (m.method) {
        for (const handler of cdp.listeners.get(m.method) ?? []) {
          handler(m.params)
        }
      }
    })

    ws.addEventListener('close', () => {
      for (const { reject } of cdp.pending.values()) {
        reject(new Error('CDP socket closed'))
      }

      cdp.pending.clear()
    })

    return cdp
  }

  /** Connect straight to a discovered target. */
  static async connect(opts) {
    const target = await discoverTarget(opts)

    return CDP.open(target.webSocketDebuggerUrl)
  }

  send(method, params = {}) {
    if (this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error(`CDP websocket is not open for ${method}`))
    }

    const id = ++this.id

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.ws.send(JSON.stringify({ id, method, params }))
    })
  }

  on(method, handler) {
    if (!this.listeners.has(method)) {
      this.listeners.set(method, [])
    }

    this.listeners.get(method).push(handler)

    return () => {
      const handlers = this.listeners.get(method)

      if (!handlers) return
      const index = handlers.indexOf(handler)

      if (index >= 0) handlers.splice(index, 1)
      if (handlers.length === 0) this.listeners.delete(method)
    }
  }

  waitFor(method, timeoutMs = 30_000) {
    return new Promise((resolve, reject) => {
      const unsubscribe = this.on(method, params => {
        clearTimeout(timeout)
        unsubscribe()
        resolve(params)
      })
      const timeout = setTimeout(() => {
        unsubscribe()
        reject(new Error(`timed out waiting for CDP event ${method}`))
      }, timeoutMs)

      timeout.unref?.()
    })
  }

  /** Evaluate an expression in the page and return its value (awaits promises). */
  async eval(expression, options = {}) {
    const r = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true,
      ...options
    })

    if (r.exceptionDetails) {
      throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text || 'eval failed')
    }

    return r.result.value
  }

  close() {
    this.ws.close()
  }
}

/** Assert the renderer has the dev-only `__PERF_DRIVE__` harness attached. */
export async function requireDriver(cdp) {
  const ok = await cdp.eval('!!(window.__PERF_DRIVE__ && window.__PERF_DRIVE__.stream)')

  if (!ok) {
    throw new Error(
      '__PERF_DRIVE__ not on window. The perf harness needs a DEV renderer ' +
        '(perf-probe.tsx is excluded from production builds). Launch with `npm run perf:serve`.'
    )
  }
}

/** Type real key events into the composer, one char at a time, at `cps` chars/sec. */
export async function typeIntoComposer(cdp, text, { cps = 15 } = {}) {
  await cdp.eval(`(() => {
    const el = document.querySelector(${JSON.stringify(SELECTORS.composer)})
    if (!el) return false
    el.focus()
    const range = document.createRange()
    range.selectNodeContents(el)
    range.collapse(false)
    const sel = window.getSelection()
    sel.removeAllRanges()
    sel.addRange(range)
    return true
  })()`)

  const intervalMs = Math.max(1, Math.round(1000 / cps))

  for (const ch of text) {
    await cdp.send('Input.dispatchKeyEvent', { type: 'char', text: ch, unmodifiedText: ch })
    await sleep(intervalMs)
  }
}

/**
 * Run `body()` while a V8 CPU profile is recording. Returns
 * `{ result, profile }`; the caller decides whether to write the .cpuprofile.
 */
export async function withCpuProfile(cdp, body, { samplingIntervalUs = 100 } = {}) {
  await cdp.send('Profiler.enable')
  await cdp.send('Profiler.setSamplingInterval', { interval: samplingIntervalUs })
  await cdp.send('Profiler.start')

  let result
  let stopped

  try {
    result = await body()
  } finally {
    // Always stop so a scenario error can't leave the profiler running.
    stopped = await cdp.send('Profiler.stop')
  }

  return { result, profile: stopped.profile }
}

export { sleep }
