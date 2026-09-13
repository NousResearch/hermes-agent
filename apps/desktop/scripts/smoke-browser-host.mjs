#!/usr/bin/env node

import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawn, spawnSync } from 'node:child_process'
import process from 'node:process'

import { CDP, sleep } from './perf/lib/cdp.mjs'

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
    await sleep(100)
  }
  throw new Error(`Chromium DevTools endpoint did not become ready: ${lastError || 'timeout'}\n${stderrTail()}`)
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
    'getAgentRoster',
    'getConnection',
    'getConnectionFor',
    'getGatewayWsUrlFor',
    'getProfileRoutes',
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
    authRequired: window.__HERMES_AUTH_REQUIRED__ === true,
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
    while (!output && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 50))
    }
    const cwd = await terminal.cwd(session.id)
    stop()
    await terminal.dispose(session.id)
    const fatalOutput = [
      'Chat unavailable:',
      'Chat failed to start:',
      'Terminal unavailable:',
      'Terminal failed to start:',
      'Pseudo-terminal support is unavailable',
      'ModuleNotFoundError:',
      'Traceback (most recent call last)'
    ].find(marker => output.includes(marker))
    const hostShellStarted = session.shell !== 'hermes-tui' && Boolean(output)
    return {
      cwd,
      error: fatalOutput
        ? 'host shell startup error: ' + fatalOutput
        : hostShellStarted
          ? undefined
          : 'host shell did not produce PTY output',
      ok: hostShellStarted && !fatalOutput,
      outputTail: output.slice(-800),
      shell: session.shell,
      skipped: false
    }
  } catch (error) {
    if (session?.id) await terminal.dispose(session.id).catch(() => undefined)
    return { error: error instanceof Error ? error.message : String(error), ok: false, skipped: false }
  }
})()`

const htmlSandboxExpression = `(async () => {
  const bridge = window.hermesDesktop
  if (!bridge?.saveImageBuffer) return { error: 'buffer bridge missing', ok: false }
  const marker = 'hermesSandboxEscape'
  delete document.documentElement.dataset[marker]
  const attack = '<script>try{top.document.documentElement.dataset.' + marker + '=\"escaped\"}catch(_error){}<\/script>'
  const url = await bridge.saveImageBuffer(new TextEncoder().encode(attack), '.html')
  const frame = document.createElement('iframe')
  frame.hidden = true
  const loaded = new Promise(resolve => {
    frame.addEventListener('load', resolve, { once: true })
    setTimeout(resolve, 2000)
  })
  frame.src = url
  document.body.appendChild(frame)
  await loaded
  await new Promise(resolve => setTimeout(resolve, 200))
  const escaped = document.documentElement.dataset[marker] === 'escaped'
  frame.remove()
  URL.revokeObjectURL(url)
  delete document.documentElement.dataset[marker]
  return { escaped, ok: !escaped }
})()`

const layoutExpression = `(async () => {
  const visible = element => {
    if (!(element instanceof HTMLElement)) return false
    const rect = element.getBoundingClientRect()
    const style = getComputedStyle(element)
    return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden'
  }
  const button = label => [...document.querySelectorAll('button')].find(item =>
    item.getAttribute('aria-label') === label || item.textContent?.trim() === label
  )
  const editor = button('Layout editor')
  if (!editor) return { error: 'layout editor button missing', ok: false }
  editor.click()
  await new Promise(resolve => setTimeout(resolve, 100))
  const quad = button('Quad')
  if (!quad) return { error: 'Quad preset button missing', ok: false }
  quad.click()
  await new Promise(resolve => setTimeout(resolve, 100))
  const done = button('Done')
  if (!done) return { error: 'Done button missing', ok: false }
  done.click()
  const read = () => {
    const files = document.querySelector('[data-tree-tab="files"]')
    const review = document.querySelector('[aria-label="Review"]')
    const terminal = document.querySelector('[data-tree-tab="terminal"]')
    return {
      editModeClosed: !button('Done'),
      files: visible(files),
      review: visible(review),
      terminal: visible(terminal)
    }
  }
  const deadline = Date.now() + 3000
  let state = read()
  while (!(state.editModeClosed && state.files && state.review && state.terminal) && Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, 50))
    state = read()
  }
  const activePreset = localStorage.getItem('hermes.desktop.layoutPreset.active')
  return {
    activePreset,
    ...state,
    ok: activePreset === 'quad' && state.editModeClosed && state.files && state.review && state.terminal
  }
})()`

const mobileViewportExpression = `(() => {
  const visibleBottom = document.documentElement.clientHeight
  const bottom = selector => document.querySelector(selector)?.getBoundingClientRect().bottom ?? null
  const shellBottom = bottom('[data-contrib-shell]')
  const sidebarBottom = bottom('[data-slot="sidebar-wrapper"]')
  const composerBottom = bottom('[data-slot="composer-dock"]')
  const contained = value => typeof value === 'number' && value <= visibleBottom + 1
  return {
    composerBottom,
    ok: contained(shellBottom) && contained(sidebarBottom) && contained(composerBottom),
    shellBottom,
    sidebarBottom,
    visibleBottom
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
    browserControl = await CDP.open(browserVersion.webSocketDebuggerUrl)
  }

  for (const viewport of [
    { width: 844, height: 390, mobile: true, smallViewportHeightDifference: 120 },
    { width: 1280, height: 800 },
    { width: 390, height: 844 },
    { width: 320, height: 568 }
  ]) {
    const target = await createTarget(debugPort)
    const client = await CDP.open(target.webSocketDebuggerUrl)
    const evaluate = expression => client.eval(expression, { userGesture: true })

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
      mobile: viewport.mobile ?? false
    })
    const domReady = client.waitFor('Page.domContentEventFired', 30_000)
    await client.send('Page.navigate', { url })
    await domReady
    await sleep(4_000)

    // Mobile browser controls shrink the small viewport without changing the
    // legacy 100vh size. Apply this after navigation so Chromium recalculates
    // the live top-level frame.
    if (viewport.smallViewportHeightDifference) {
      await client.send('Emulation.setSmallViewportHeightDifferenceOverride', {
        difference: viewport.smallViewportHeightDifference
      })
      await sleep(100)
    }

    const state = await evaluate(stateExpression)
    let htmlSandboxState = { skipped: true }
    let layoutState = { skipped: true }
    let mobileViewportState = { skipped: true }
    let terminalState = { skipped: true }
    if (viewport.width === 1280) htmlSandboxState = await evaluate(htmlSandboxExpression)
    if (viewport.width === 1280) layoutState = await evaluate(layoutExpression)
    if (viewport.smallViewportHeightDifference) mobileViewportState = await evaluate(mobileViewportExpression)
    if (requireTerminal && viewport.width === 1280) terminalState = await evaluate(terminalExpression)

    const failures = []
    if (httpStatus === null || httpStatus >= 400) failures.push(`HTTP ${httpStatus ?? 'no response'}`)
    if (state.host !== 'browser' || !state.bridge) failures.push('browser Desktop bridge did not install')
    if (!state.sessionToken && !state.authRequired) failures.push('browser auth bootstrap was not injected')
    if (!state.requiredBridgeMethods) failures.push('required browser Desktop bridge methods are missing')
    if (state.rootError) failures.push('renderer reached its root error boundary')
    if (!allowGatewayFailure && state.desktopBootFailed)
      failures.push('Desktop could not connect to the Hermes gateway')
    if (viewport.width === 1280 && !layoutState.ok) {
      failures.push(`configured layout reverted after leaving edit mode: ${JSON.stringify(layoutState)}`)
    }
    if (viewport.width === 1280 && !htmlSandboxState.ok) {
      failures.push(`browser HTML artifact escaped its sandbox: ${JSON.stringify(htmlSandboxState)}`)
    }
    if (
      viewport.smallViewportHeightDifference &&
      mobileViewportState.visibleBottom !== viewport.height - viewport.smallViewportHeightDifference
    ) {
      failures.push(`mobile viewport emulation did not shrink the visible page: ${JSON.stringify(mobileViewportState)}`)
    }
    if (viewport.smallViewportHeightDifference && !mobileViewportState.ok) {
      failures.push(`mobile browser controls cover the app shell: ${JSON.stringify(mobileViewportState)}`)
    }
    if (requireTerminal && viewport.width === 1280 && !terminalState.ok) {
      failures.push(
        `browser Desktop host terminal failed: ${terminalState.error || terminalState.outputTail || 'marker missing'}`
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

    console.log(
      JSON.stringify(
        { failures, htmlSandboxState, http: httpStatus, layoutState, mobileViewportState, state, terminalState, viewport },
        null,
        2
      )
    )
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
  // temporary user-data-dir. Chromium exposed a real ENOTEMPTY race when
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
    await Promise.race([new Promise(resolve => browser.once('exit', resolve)), sleep(timeoutMs)])
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
