#!/usr/bin/env node

import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawn, spawnSync } from 'node:child_process'
import process from 'node:process'

const url = process.env.HERMES_BROWSER_HOST_URL || process.argv[2] || 'http://127.0.0.1:9119/'
const allowGatewayFailure = process.env.HERMES_BROWSER_ALLOW_GATEWAY_FAILURE === '1'
const requireTerminal = process.env.HERMES_BROWSER_REQUIRE_TERMINAL === '1'

function findExecutable() {
  const explicit = process.env.HERMES_BROWSER_EXECUTABLE?.trim()
  if (explicit) return explicit

  const candidates =
    process.platform === 'win32'
      ? [
          'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
          'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
          'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe'
        ]
      : ['chromium-browser', 'chromium', 'google-chrome', 'google-chrome-stable']

  for (const candidate of candidates) {
    if (candidate.includes('\\') && existsSync(candidate)) return candidate
    if (!candidate.includes('\\')) {
      const found = spawnSync('sh', ['-lc', `command -v ${candidate}`], { encoding: 'utf8' })
      const path = found.status === 0 ? found.stdout.trim() : ''
      if (path) return path
    }
  }

  throw new Error('No Chromium-compatible browser found; set HERMES_BROWSER_EXECUTABLE')
}

const ignoredConsoleError = text =>
  text.includes("Blocked call to navigator.vibrate because user hasn't tapped") ||
  (allowGatewayFailure && text.includes('WebSocket connection to'))

const delay = ms => new Promise(resolve => setTimeout(resolve, ms))

async function freePort() {
  const server = createServer()
  await new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })
  const address = server.address()
  const port = typeof address === 'object' && address ? address.port : 0
  await new Promise(resolve => server.close(resolve))
  if (!port) throw new Error('could not allocate a Chromium debugging port')
  return port
}

async function waitForJson(endpoint, browser, stderrTail, timeoutMs = 15_000) {
  const deadline = Date.now() + timeoutMs
  let lastError
  while (Date.now() < deadline) {
    if (browser.exitCode !== null) {
      throw new Error(`Chromium exited during startup (${browser.exitCode})\n${stderrTail()}`)
    }
    try {
      const response = await fetch(endpoint)
      if (response.ok) return await response.json()
      lastError = new Error(`HTTP ${response.status}`)
    } catch (error) {
      lastError = error
    }
    await delay(100)
  }
  throw new Error(`Chromium DevTools endpoint did not become ready: ${lastError || 'timeout'}\n${stderrTail()}`)
}

class CdpClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl
    this.socket = null
    this.nextId = 1
    this.pending = new Map()
    this.listeners = new Map()
    this.waiters = new Map()
  }

  async connect() {
    this.socket = new WebSocket(this.wsUrl)
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('CDP websocket open timeout')), 10_000)
      this.socket.addEventListener(
        'open',
        () => {
          clearTimeout(timer)
          resolve()
        },
        { once: true }
      )
      this.socket.addEventListener(
        'error',
        () => {
          clearTimeout(timer)
          reject(new Error('CDP websocket failed to open'))
        },
        { once: true }
      )
    })
    this.socket.addEventListener('message', event => {
      void this.#handleMessage(event.data)
    })
    this.socket.addEventListener('close', () => {
      for (const { reject } of this.pending.values()) reject(new Error('CDP websocket closed'))
      this.pending.clear()
    })
  }

  async #handleMessage(data) {
    let text
    if (typeof data === 'string') text = data
    else if (data instanceof Blob) text = await data.text()
    else text = Buffer.from(data).toString('utf8')

    const message = JSON.parse(text)
    if (message.id) {
      const pending = this.pending.get(message.id)
      if (!pending) return
      this.pending.delete(message.id)
      if (message.error) pending.reject(new Error(`${message.error.message} (${message.error.code})`))
      else pending.resolve(message.result || {})
      return
    }

    if (!message.method) return
    for (const listener of this.listeners.get(message.method) || []) {
      try {
        listener(message.params || {})
      } catch {
        /* observer errors must not break CDP */
      }
    }
    const waiters = this.waiters.get(message.method)
    if (waiters?.length) {
      const waiter = waiters.shift()
      waiter.resolve(message.params || {})
      if (!waiters.length) this.waiters.delete(message.method)
    }
  }

  on(method, listener) {
    const listeners = this.listeners.get(method) || []
    listeners.push(listener)
    this.listeners.set(method, listeners)
    return () => {
      const current = this.listeners.get(method) || []
      this.listeners.set(
        method,
        current.filter(item => item !== listener)
      )
    }
  }

  waitFor(method, timeoutMs = 30_000) {
    return new Promise((resolve, reject) => {
      const waiters = this.waiters.get(method) || []
      const waiter = { resolve, reject }
      waiters.push(waiter)
      this.waiters.set(method, waiters)
      setTimeout(() => {
        const current = this.waiters.get(method) || []
        const index = current.indexOf(waiter)
        if (index >= 0) current.splice(index, 1)
        if (!current.length) this.waiters.delete(method)
        reject(new Error(`timed out waiting for CDP event ${method}`))
      }, timeoutMs).unref?.()
    })
  }

  send(method, params = {}) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error(`CDP websocket is not open for ${method}`))
    }
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.socket.send(JSON.stringify({ id, method, params }))
    })
  }

  async evaluate(expression) {
    const result = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true
    })
    if (result.exceptionDetails) {
      const description =
        result.exceptionDetails.exception?.description || result.exceptionDetails.text || 'evaluation failed'
      throw new Error(description)
    }
    return result.result?.value
  }

  close() {
    try {
      this.socket?.close()
    } catch {
      /* best effort */
    }
  }
}

async function createTarget(debugPort) {
  const endpoint = `http://127.0.0.1:${debugPort}/json/new?${encodeURIComponent('about:blank')}`
  const response = await fetch(endpoint, { method: 'PUT' })
  if (!response.ok) throw new Error(`failed to create Chromium target: HTTP ${response.status}`)
  return await response.json()
}

function consoleText(params) {
  return (params.args || [])
    .map(arg => {
      if (Object.hasOwn(arg, 'value')) return typeof arg.value === 'string' ? arg.value : JSON.stringify(arg.value)
      return arg.description || arg.type || ''
    })
    .join(' ')
}

const stateExpression = `(() => {
  const requiredBridgeMethods = [
    'api',
    'getConnection',
    'getRecentLogs',
    'normalizePreviewTarget',
    'readDir',
    'readFileText',
    'revealLogs',
    'saveClipboardImage',
    'watchPreviewFile'
  ]
  const text = document.body.innerText || ''
  return {
    bodyScrollWidth: document.body.scrollWidth,
    bridge: Boolean(window.hermesDesktop),
    clientWidth: document.documentElement.clientWidth,
    desktopBootFailed: text.includes('Desktop boot failed'),
    host: document.documentElement.dataset.hermesDesktopHost || '',
    requiredBridgeMethods: requiredBridgeMethods.every(key => typeof window.hermesDesktop?.[key] === 'function'),
    rootError: text.includes('Something broke in the interface'),
    scrollWidth: document.documentElement.scrollWidth,
    sessionToken: Boolean(window.__HERMES_SESSION_TOKEN__),
    title: document.title
  }
})()`

const terminalExpression = `(async () => {
  const terminal = window.hermesDesktop?.terminal
  if (!terminal) return { error: 'terminal bridge missing', ok: false, skipped: false }
  let session
  try {
    session = await terminal.start({ cols: 48, rows: 16 })
    let output = ''
    const stop = terminal.onData(session.id, chunk => { output += chunk })
    await terminal.resize(session.id, { cols: 52, rows: 18 })
    const deadline = Date.now() + 8000
    while (!output.includes('Hermes') && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 50))
    }
    const cwd = await terminal.cwd(session.id)
    stop()
    await terminal.dispose(session.id)
    const fatalOutput = [
      'Chat unavailable:',
      'Chat failed to start:',
      'Pseudo-terminal support is unavailable',
      'ModuleNotFoundError:',
      'Traceback (most recent call last)'
    ].find(marker => output.includes(marker))
    const paintedHermesFrame = output.includes('Hermes')
    return {
      cwd,
      error: fatalOutput
        ? 'Hermes TUI startup error: ' + fatalOutput
        : paintedHermesFrame
          ? undefined
          : 'Hermes TUI did not paint a branded frame',
      ok: session.shell === 'hermes-tui' && paintedHermesFrame && !fatalOutput,
      outputTail: output.slice(-800),
      shell: session.shell,
      skipped: false
    }
  } catch (error) {
    if (session?.id) await terminal.dispose(session.id).catch(() => undefined)
    return { error: error instanceof Error ? error.message : String(error), ok: false, skipped: false }
  }
})()`

const executable = findExecutable()
const debugPort = await freePort()
const profile = mkdtempSync(join(tmpdir(), 'hermes-browser-smoke-'))
const stderrChunks = []
const browser = spawn(
  executable,
  [
    '--headless=new',
    '--disable-dev-shm-usage',
    '--remote-debugging-address=127.0.0.1',
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${profile}`,
    'about:blank'
  ],
  { stdio: ['ignore', 'ignore', 'pipe'] }
)
browser.stderr?.on('data', chunk => {
  stderrChunks.push(Buffer.from(chunk))
  while (stderrChunks.reduce((sum, item) => sum + item.length, 0) > 24_000) stderrChunks.shift()
})
const stderrTail = () => Buffer.concat(stderrChunks).toString('utf8').slice(-24_000)

let browserControl = null

try {
  const browserVersion = await waitForJson(`http://127.0.0.1:${debugPort}/json/version`, browser, stderrTail)
  if (browserVersion?.webSocketDebuggerUrl) {
    browserControl = new CdpClient(browserVersion.webSocketDebuggerUrl)
    await browserControl.connect()
  }

  for (const viewport of [
    { width: 390, height: 844 },
    { width: 320, height: 568 }
  ]) {
    const target = await createTarget(debugPort)
    const client = new CdpClient(target.webSocketDebuggerUrl)
    await client.connect()

    const consoleErrors = []
    const pageErrors = []
    const failedRequests = []
    const requestUrls = new Map()
    let httpStatus = null

    client.on('Runtime.consoleAPICalled', params => {
      if (params.type !== 'error') return
      const text = consoleText(params)
      if (!ignoredConsoleError(text)) consoleErrors.push(text)
    })
    client.on('Runtime.exceptionThrown', params => {
      const error = params.exceptionDetails?.exception?.description || params.exceptionDetails?.text || 'page exception'
      pageErrors.push(error)
    })
    client.on('Network.requestWillBeSent', params => {
      requestUrls.set(params.requestId, params.request?.url || '')
    })
    client.on('Network.responseReceived', params => {
      if (params.type === 'Document' && params.response?.url?.startsWith(url)) httpStatus = params.response.status
    })
    client.on('Network.loadingFailed', params => {
      const requestUrl = requestUrls.get(params.requestId) || ''
      if (allowGatewayFailure && requestUrl.includes('/api/ws')) return
      failedRequests.push(`${requestUrl || params.requestId} (${params.errorText || 'request failed'})`)
    })

    await Promise.all([client.send('Page.enable'), client.send('Runtime.enable'), client.send('Network.enable')])
    await client.send('Emulation.setDeviceMetricsOverride', {
      width: viewport.width,
      height: viewport.height,
      deviceScaleFactor: 1,
      mobile: false
    })
    const domReady = client.waitFor('Page.domContentEventFired', 30_000)
    await client.send('Page.navigate', { url })
    await domReady
    await delay(4_000)

    const state = await client.evaluate(stateExpression)
    let terminalState = { skipped: true }
    if (requireTerminal && viewport.width === 390) terminalState = await client.evaluate(terminalExpression)

    const failures = []
    if (httpStatus === null || httpStatus >= 400) failures.push(`HTTP ${httpStatus ?? 'no response'}`)
    if (state.host !== 'browser' || !state.bridge) failures.push('browser Desktop bridge did not install')
    if (!state.sessionToken) failures.push('loopback session token was not injected')
    if (!state.requiredBridgeMethods) failures.push('required browser Desktop bridge methods are missing')
    if (state.rootError) failures.push('renderer reached its root error boundary')
    if (!allowGatewayFailure && state.desktopBootFailed)
      failures.push('Desktop could not connect to the Hermes gateway')
    if (requireTerminal && viewport.width === 390 && !terminalState.ok) {
      failures.push(
        `browser Desktop Hermes TUI terminal failed: ${terminalState.error || terminalState.outputTail || 'marker missing'}`
      )
    }
    if (state.scrollWidth > state.clientWidth || state.bodyScrollWidth > state.clientWidth) {
      failures.push(
        `horizontal overflow: viewport=${state.clientWidth}, html=${state.scrollWidth}, body=${state.bodyScrollWidth}`
      )
    }
    failures.push(...pageErrors.map(error => `pageerror: ${error}`))
    failures.push(...consoleErrors.map(error => `console: ${error}`))
    failures.push(...failedRequests.map(error => `request: ${error}`))

    console.log(JSON.stringify({ failures, http: httpStatus, state, terminalState, viewport }, null, 2))
    try {
      await client.send('Page.close')
    } catch {
      /* target may already be closing */
    }
    client.close()

    if (failures.length) {
      process.exitCode = 1
      break
    }
  }
} finally {
  // Ask Chromium to shut down through its own browser-level CDP endpoint first.
  // This lets profile writers and helper processes flush before we remove the
  // temporary user-data-dir. Native Termux exposed a real ENOTEMPTY race when
  // the smoke killed Chromium and immediately called rmSync despite all UX
  // assertions already passing.
  if (browserControl) {
    try {
      await browserControl.send('Browser.close')
    } catch {
      /* Chromium may close the control socket before acknowledging Browser.close */
    }
    browserControl.close()
  }

  const hasExited = () => browser.exitCode !== null || browser.signalCode !== null
  const waitForExit = async timeoutMs => {
    if (hasExited()) return true
    await Promise.race([new Promise(resolve => browser.once('exit', resolve)), delay(timeoutMs)])
    return hasExited()
  }

  if (!(await waitForExit(3_000))) {
    browser.kill('SIGTERM')
    if (!(await waitForExit(3_000))) {
      browser.kill('SIGKILL')
      await waitForExit(3_000)
    }
  }

  rmSync(profile, { recursive: true, force: true, maxRetries: 12, retryDelay: 125 })
}
