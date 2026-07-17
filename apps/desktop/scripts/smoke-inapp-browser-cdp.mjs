#!/usr/bin/env node

import { mkdir, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

function parseArgs(argv) {
  const options = {
    click: false,
    outDir: '/tmp/hermes-inapp-browser-cdp-smoke',
    port: 9333,
    selector: 'body',
    targetUrl: 'https://example.com/'
  }

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (arg === '--click') options.click = true
    else if (['--out-dir', '--port', '--selector', '--target-url'].includes(arg)) {
      const value = argv[index + 1]
      if (!value || value.startsWith('--')) throw new Error(`${arg} requires a value`)
      if (arg === '--out-dir') options.outDir = value
      if (arg === '--port') options.port = Number(value)
      if (arg === '--selector') options.selector = value
      if (arg === '--target-url') options.targetUrl = value
      index += 1
    } else if (arg === '--help') {
      console.log('Usage: node scripts/smoke-inapp-browser-cdp.mjs [--port 9333] [--target-url URL] [--selector CSS] [--click] [--out-dir DIR]')
      process.exit(0)
    } else throw new Error(`Unknown option: ${arg}`)
  }

  if (!Number.isSafeInteger(options.port) || options.port < 1 || options.port > 65535) throw new Error('--port must be an integer from 1 to 65535')
  if (!options.selector.trim()) throw new Error('--selector must not be empty')
  if (!/^https?:\/\//.test(options.targetUrl)) throw new Error('--target-url must be an HTTP(S) URL prefix')
  return options
}

class CdpClient {
  constructor(socket) {
    this.socket = socket
    this.nextId = 0
    this.pending = new Map()
    socket.addEventListener('message', event => {
      const message = JSON.parse(String(event.data))
      if (message.id == null) return
      const pending = this.pending.get(message.id)
      if (!pending) return
      this.pending.delete(message.id)
      clearTimeout(pending.timeout)
      if (message.error) pending.reject(new Error(`${pending.method}: ${message.error.message}`))
      else pending.resolve(message.result)
    })
    socket.addEventListener('close', () => {
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timeout)
        pending.reject(new Error(`CDP socket closed during ${pending.method}`))
      }
      this.pending.clear()
    })
  }

  static open(url) {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(url)
      const timeout = setTimeout(() => reject(new Error(`Timed out connecting to ${url}`)), 10_000)
      socket.addEventListener('open', () => {
        clearTimeout(timeout)
        resolve(new CdpClient(socket))
      }, { once: true })
      socket.addEventListener('error', () => {
        clearTimeout(timeout)
        reject(new Error(`Could not connect to ${url}`))
      }, { once: true })
    })
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.nextId
      const timeout = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`${method} timed out`))
      }, 15_000)
      this.pending.set(id, { method, reject, resolve, timeout })
      this.socket.send(JSON.stringify({ id, method, params }))
    })
  }

  close() {
    this.socket.close()
  }
}

function runtimeValue(result) {
  if (result?.exceptionDetails) throw new Error(result.exceptionDetails.text || 'Runtime evaluation failed')
  return result?.result?.value
}

async function evaluate(client, expression, { awaitPromise = false } = {}) {
  return runtimeValue(await client.send('Runtime.evaluate', { expression, awaitPromise, returnByValue: true }))
}

function pngDimensions(buffer) {
  if (buffer.length < 24 || buffer.toString('ascii', 1, 4) !== 'PNG') throw new Error('CDP screenshot is not a PNG')
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) }
}

function rectFromQuad(quad) {
  if (!Array.isArray(quad) || quad.length !== 8) return null
  const xs = [quad[0], quad[2], quad[4], quad[6]]
  const ys = [quad[1], quad[3], quad[5], quad[7]]
  const left = Math.min(...xs)
  const right = Math.max(...xs)
  const top = Math.min(...ys)
  const bottom = Math.max(...ys)
  return { bottom, height: bottom - top, left, right, top, width: right - left, x: left, y: top }
}

async function discoverHostTarget(targets, guestTarget, targetUrl) {
  for (const target of targets) {
    if (target.id === guestTarget.id || target.type !== 'page' || !target.webSocketDebuggerUrl) continue
    let client
    try {
      client = await CdpClient.open(target.webSocketDebuggerUrl)
      const probe = await evaluate(client, `(() => {
        const views = Array.from(document.querySelectorAll('webview[partition="persist:hermes-browser"]'))
        return {
          hasBrowserBridge: Boolean(window.hermesDesktop?.browser?.capture && window.hermesDesktop?.browser?.cdp),
          matchingWebviews: views.filter(view => String(view.getURL?.() || view.src || '').startsWith(${JSON.stringify(targetUrl)})).length,
          webviewCount: views.length
        }
      })()`)
      if (probe?.hasBrowserBridge && probe.webviewCount > 0) return { client, probe, target }
    } catch {
      // A non-Hermes renderer is not the host target.
    }
    client?.close()
  }
  return null
}

async function hostReadback(host, targetUrl) {
  if (!host) return null
  return evaluate(host.client, `(async () => {
    const views = Array.from(document.querySelectorAll('webview[partition="persist:hermes-browser"]'))
    const view = views.find(candidate => String(candidate.getURL?.() || candidate.src || '').startsWith(${JSON.stringify(targetUrl)}))
    if (!view) return { found: false, webviewCount: views.length }
    const rect = view.getBoundingClientRect()
    const guestWebContentsId = view.getWebContentsId?.()
    const browser = window.hermesDesktop?.browser
    const [capture, guest] = await Promise.all([
      browser?.capture && Number.isInteger(guestWebContentsId) ? browser.capture(guestWebContentsId) : null,
      browser?.metrics && Number.isInteger(guestWebContentsId) ? browser.metrics(guestWebContentsId) : null
    ])
    const style = getComputedStyle(view.parentElement || view)
    return {
      found: true,
      guestWebContentsId,
      src: String(view.getURL?.() || view.src || ''),
      elementRect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height, top: rect.top, right: rect.right, bottom: rect.bottom, left: rect.left },
      parentTransform: style.transform,
      renderer: {
        innerWidth,
        innerHeight,
        outerWidth,
        outerHeight,
        devicePixelRatio,
        visualViewport: visualViewport ? { width: visualViewport.width, height: visualViewport.height, scale: visualViewport.scale } : null
      },
      capturePage: capture ? { width: capture.width, height: capture.height, createdAt: capture.createdAt } : null,
      guest
    }
  })()`, { awaitPromise: true })
}

function safeElementState() {
  return `(() => {
    const element = document.querySelector(${JSON.stringify(globalThis.__smokeSelector)})
    const focused = document.activeElement
    const stateFor = value => value ? {
      tag: value.tagName,
      id: value.id || null,
      className: typeof value.className === 'string' ? value.className : null,
      ariaExpanded: value.getAttribute?.('aria-expanded'),
      ariaSelected: value.getAttribute?.('aria-selected'),
      dataState: value.getAttribute?.('data-state'),
      checked: typeof value.checked === 'boolean' ? value.checked : null,
      disabled: typeof value.disabled === 'boolean' ? value.disabled : null
    } : null
    const rect = element?.getBoundingClientRect()
    return {
      url: location.href,
      title: document.title,
      focused: stateFor(focused),
      element: stateFor(element),
      elementRect: rect ? { x: rect.x, y: rect.y, width: rect.width, height: rect.height, top: rect.top, right: rect.right, bottom: rect.bottom, left: rect.left } : null
    }
  })()`
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  globalThis.__smokeSelector = options.selector
  const endpoint = `http://127.0.0.1:${options.port}`
  const targets = await (await fetch(`${endpoint}/json/list`)).json()
  const candidates = targets.filter(target => ['page', 'webview'].includes(target.type) && String(target.url || '').startsWith(options.targetUrl))
  if (candidates.length !== 1) throw new Error(`Expected exactly one guest target for ${options.targetUrl}; found ${candidates.length}`)
  const guestTarget = candidates[0]
  const guest = await CdpClient.open(guestTarget.webSocketDebuggerUrl)
  const host = await discoverHostTarget(targets, guestTarget, options.targetUrl)

  try {
    const before = await evaluate(guest, `(() => ({
      url: location.href,
      title: document.title,
      innerWidth,
      innerHeight,
      outerWidth,
      outerHeight,
      devicePixelRatio,
      scrollX,
      scrollY,
      visualViewport: visualViewport ? { width: visualViewport.width, height: visualViewport.height, scale: visualViewport.scale } : null
    }))()`)
    const documentResult = await guest.send('DOM.getDocument', { depth: 0, pierce: true })
    const query = await guest.send('DOM.querySelector', { nodeId: documentResult.root.nodeId, selector: options.selector })
    if (!query.nodeId) throw new Error(`Selector did not match: ${options.selector}`)

    let boxModel = null
    try {
      boxModel = await guest.send('DOM.getBoxModel', { nodeId: query.nodeId })
    } catch {
      // Runtime rect below remains authoritative when a node has no box model.
    }
    const beforeState = await evaluate(guest, safeElementState())
    const boxRect = rectFromQuad(boxModel?.model?.content || boxModel?.model?.border)
    const cssRect = boxRect || beforeState.elementRect
    if (!cssRect || cssRect.width <= 0 || cssRect.height <= 0) throw new Error(`Selector has no positive CSS box: ${options.selector}`)
    const cssCenter = { x: cssRect.left + cssRect.width / 2, y: cssRect.top + cssRect.height / 2 }

    const layout = await guest.send('Page.getLayoutMetrics')
    const screenshotResult = await guest.send('Page.captureScreenshot', { captureBeyondViewport: false, format: 'png', fromSurface: true })
    const screenshot = Buffer.from(screenshotResult.data, 'base64')
    const screenshotSize = pngDimensions(screenshot)

    if (options.click) {
      await guest.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: cssCenter.x, y: cssCenter.y })
      await guest.send('Input.dispatchMouseEvent', { button: 'left', clickCount: 1, type: 'mousePressed', x: cssCenter.x, y: cssCenter.y })
      await guest.send('Input.dispatchMouseEvent', { button: 'left', clickCount: 1, type: 'mouseReleased', x: cssCenter.x, y: cssCenter.y })
      await new Promise(resolve => setTimeout(resolve, 750))
    }

    const after = await evaluate(guest, safeElementState())
    const hostState = await hostReadback(host, options.targetUrl)
    const cssViewport = layout.cssVisualViewport || layout.visualViewport
    const cssWidth = cssViewport?.clientWidth || before.innerWidth
    const cssHeight = cssViewport?.clientHeight || before.innerHeight
    const displayScale = hostState?.elementRect && cssWidth && cssHeight
      ? { x: hostState.elementRect.width / cssWidth, y: hostState.elementRect.height / cssHeight }
      : null

    const report = {
      ok: true,
      endpoint,
      requestedTargetUrl: options.targetUrl,
      targetInventory: targets.map(target => ({ id: target.id, title: target.title, type: target.type, url: target.url })),
      selectedGuestTarget: { id: guestTarget.id, title: guestTarget.title, type: guestTarget.type, url: guestTarget.url },
      hostRendererTarget: host ? { id: host.target.id, title: host.target.title, type: host.target.type, url: host.target.url } : null,
      dom: { selector: options.selector, documentNodeId: documentResult.root.nodeId, nodeId: query.nodeId, boxModel: boxModel?.model || null, cssRect, cssCenter },
      action: { clickDispatched: options.click, billableActionsExecuted: [] },
      readback: { before, beforeState, after, urlChanged: beforeState.url !== after.url, focusChanged: JSON.stringify(beforeState.focused) !== JSON.stringify(after.focused), elementStateChanged: JSON.stringify(beforeState.element) !== JSON.stringify(after.element) },
      pageLayoutMetrics: layout,
      screenshots: {
        cdp: { width: screenshotSize.width, height: screenshotSize.height, path: resolve(options.outDir, 'guest-cdp.png') },
        electronCapturePage: hostState?.capturePage || null
      },
      hostState,
      coordinateSpaces: {
        cssPx: { center: cssCenter, rect: cssRect, viewport: { width: cssWidth, height: cssHeight }, source: 'DOM.getBoxModel/getBoundingClientRect and Page.getLayoutMetrics' },
        cdpScreenshotPx: { width: screenshotSize.width, height: screenshotSize.height, pxPerCssX: screenshotSize.width / cssWidth, pxPerCssY: screenshotSize.height / cssHeight, source: 'Page.captureScreenshot PNG IHDR' },
        electronRendererPx: hostState?.renderer ? { ...hostState.renderer, source: 'Hermes host renderer Runtime.evaluate' } : { value: null, source: 'Host renderer target unavailable' },
        webviewElementLocalPx: hostState?.elementRect ? { rect: hostState.elementRect, displayScaleFromGuestCss: displayScale, mapping: 'displayed = guest CSS × displayScale; Fit transform changes display scale only', source: 'host webview getBoundingClientRect' } : { value: null, source: 'Host webview rect unavailable' },
        cuaScreenshotPx: { value: null, source: 'Requires cua-driver window screenshot' },
        macosPoint: { value: null, source: 'Requires macOS window bounds/CUA readback' },
        physicalDisplayPixel: { value: null, source: 'Requires display backing-scale readback' },
        screenAbsolutePoint: { value: null, source: 'Requires macOS window origin plus local-point transform' }
      },
      security: { endpointHost: '127.0.0.1', cookiesRead: false, credentialsRead: false, networkDomainUsed: false, storageDomainUsed: false }
    }

    await mkdir(resolve(options.outDir), { recursive: true })
    await Promise.all([
      writeFile(resolve(options.outDir, 'guest-cdp.png'), screenshot),
      writeFile(resolve(options.outDir, 'report.json'), `${JSON.stringify(report, null, 2)}\n`)
    ])
    console.log(JSON.stringify({ ok: true, report: resolve(options.outDir, 'report.json'), screenshot: resolve(options.outDir, 'guest-cdp.png'), targetId: guestTarget.id }))
  } finally {
    guest.close()
    host?.client.close()
  }
}

main().catch(error => {
  console.error(`smoke-inapp-browser-cdp: ${error instanceof Error ? error.message : String(error)}`)
  process.exitCode = 1
})
