import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'

import type { PreviewUblockState } from './preview-ublock'
import type { PreviewUblockPopupParent } from './preview-ublock-popup'
import type { PreviewUblockRuntime } from './preview-ublock-runtime'

interface PreviewUblockVerificationWebContents {
  executeJavaScript(code: string): Promise<unknown>
  id: number
  isDestroyed(): boolean
  once(event: 'destroyed', listener: () => void): unknown
  session?: unknown
}

interface PreviewUblockVerificationGuest extends PreviewUblockVerificationWebContents {
  isDestroyed(): boolean
  loadURL(url: string): Promise<void>
  on(event: 'destroyed' | 'did-navigate', listener: () => void): unknown
  removeListener?(event: 'destroyed' | 'did-navigate', listener: () => void): unknown
  reload?(): unknown
}

interface PreviewUblockOwnerWindow extends PreviewUblockPopupParent {
  destroy(): void
  loadURL(url: string): Promise<void>
  webContents: PreviewUblockVerificationWebContents
}

interface PreviewUblockOwnerPopupWindow {
  isDestroyed(): boolean
  loadURL(url: string): Promise<void>
  once(event: 'closed', listener: () => void): unknown
  webContents: PreviewUblockVerificationWebContents
}

export interface PreviewUblockSmokeDependencies {
  createOwnerWindow: () => PreviewUblockOwnerWindow
  findPopupWindow: (owner: PreviewUblockOwnerWindow) => PreviewUblockOwnerPopupWindow | null
  getGuest: (webContentsId: number) => PreviewUblockVerificationGuest | null
  previewSession: unknown
}

function writeRequestBlockerFixture(runtimeProbeUrl: string): string {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-preview-ublock-request-blocker-'))

  fs.writeFileSync(
    path.join(fixture, 'manifest.json'),
    JSON.stringify({ declarative_net_request: { rule_resources: [{ enabled: true, path: 'rules.json' }] } })
  )
  fs.writeFileSync(
    path.join(fixture, 'rules.json'),
    JSON.stringify([
      {
        action: { type: 'block' },
        condition: { resourceTypes: ['image'], urlFilter: `|${runtimeProbeUrl}` },
        id: 1
      }
    ])
  )

  return fixture
}

export async function runPreviewUblockSmoke(
  runtime: PreviewUblockRuntime,
  { createOwnerWindow, findPopupWindow, getGuest, previewSession }: PreviewUblockSmokeDependencies
): Promise<void> {
  let ublockState: PreviewUblockState | null = null
  let ownerWindow: PreviewUblockOwnerWindow | null = null
  let ownerServer: http.Server | null = null
  let requestBlockerFixture: string | null = null
  let runtimeProbeRequests = 0
  let previewGuestId: number | null = null
  let popupWindow: PreviewUblockOwnerPopupWindow | null = null
  let cleanupError: Error | null = null

  try {
    ublockState = await runtime.setEnabled(true)

    if (!ublockState.available || !ublockState.rulesetsReady || !ublockState.dashboardUrl || !ublockState.popupUrl) {
      throw new Error('uBlock Origin Lite did not become available')
    }

    ownerWindow = createOwnerWindow()
    ownerServer = http.createServer((request, response) => {
      response.setHeader('content-type', 'text/html; charset=utf-8')

      if (request.url === '/runtime-probe.png') {
        runtimeProbeRequests += 1
        response.end('this request should have been blocked')

        return
      }

      response.end(
        request.url === '/blocked'
          ? '<!doctype html><img src="http://000491b06a.com/blocked.png"><img src="http://127.0.0.1:9/unreachable.png">'
          : request.url === '/runtime-probe-document'
            ? '<!doctype html><img src="/runtime-probe.png">'
            : '<!doctype html><p>clear</p>'
      )
    })
    await new Promise<void>((resolve, reject) => {
      ownerServer!.once('error', reject)
      ownerServer!.listen(0, '127.0.0.1', () => resolve())
    })
    const ownerServerAddress = ownerServer.address()

    if (!ownerServerAddress || typeof ownerServerAddress === 'string') {
      throw new Error('uBlock Origin Lite owner-window server did not bind')
    }

    const blockedDocument = `http://127.0.0.1:${ownerServerAddress.port}/blocked`
    const clearDocument = `http://127.0.0.1:${ownerServerAddress.port}/clear`
    const runtimeProbeDocument = `http://127.0.0.1:${ownerServerAddress.port}/runtime-probe-document`
    const runtimeProbeUrl = `http://127.0.0.1:${ownerServerAddress.port}/runtime-probe.png`
    const hostDocument = `data:text/html,${encodeURIComponent(
      '<!doctype html><webview id="preview-guest" partition="persist:hermes-preview" src="about:blank" style="width:1px;height:1px"></webview>'
    )}`
    await ownerWindow.loadURL(hostDocument)

    for (let attempt = 0; attempt < 100; attempt += 1) {
      const guestId = await ownerWindow.webContents.executeJavaScript(
        `document.querySelector('#preview-guest')?.getWebContentsId?.() ?? null`
      )

      if (typeof guestId === 'number') {
        previewGuestId = guestId
        break
      }

      await new Promise(resolve => setTimeout(resolve, 50))
    }

    const guest = previewGuestId === null ? null : getGuest(previewGuestId)

    if (!guest || guest.isDestroyed() || !runtime.registerGuest(guest.id, ownerWindow.webContents.id, guest)) {
      throw new Error('uBlock Origin Lite verification guest could not be registered')
    }

    await guest.loadURL(blockedDocument)
    await guest.loadURL(clearDocument)

    await runtime.openPopup(ownerWindow)
    popupWindow = findPopupWindow(ownerWindow)

    if (!popupWindow || popupWindow.isDestroyed() || popupWindow.webContents.session !== previewSession) {
      throw new Error('uBlock Origin Lite popup window was not created in the Preview session')
    }

    const popupExtensionId = await popupWindow.webContents.executeJavaScript('chrome.runtime.id')

    if (popupExtensionId !== ublockState.extensionId) {
      throw new Error('uBlock Origin Lite popup could not access the extension runtime')
    }

    requestBlockerFixture = writeRequestBlockerFixture(runtimeProbeUrl)

    if (!runtime.loadRules(requestBlockerFixture)) {
      throw new Error('the Preview request blocker could not load its local verification rule')
    }

    await guest.loadURL(runtimeProbeDocument)
    await new Promise(resolve => setTimeout(resolve, 250))

    if (
      runtimeProbeRequests !== 0 ||
      runtime.getRequestBlockerBlockedRequestCount() === 0 ||
      runtime.getBlockedRequestCount(guest.id) === 0
    ) {
      throw new Error('the Preview request blocker did not cancel and count a webview request')
    }

    const permissionProbe = await popupWindow.webContents.executeJavaScript(
      `(async () => {
        const request = chrome.permissions?.request
        if (typeof request !== 'function') {
          return { available: false, coveredOrigin: false, unsupportedPermissionRejected: false }
        }
        const coveredOrigin = await request({ origins: ['<all_urls>'] })
        let unsupportedPermissionRejected = false
        try {
          await request({ origins: ['<all_urls>'], permissions: ['tabs'] })
        } catch {
          unsupportedPermissionRejected = true
        }
        return { available: true, coveredOrigin, unsupportedPermissionRejected }
      })()`
    )

    if (
      !(permissionProbe as { available?: unknown })?.available ||
      (permissionProbe as { coveredOrigin?: unknown }).coveredOrigin !== true ||
      !(permissionProbe as { unsupportedPermissionRejected?: unknown })?.unsupportedPermissionRejected
    ) {
      throw new Error('uBlock Origin Lite popup permission compatibility is incomplete')
    }

    const completeLevel = await popupWindow.webContents.executeJavaScript(
      `(async () => chrome.runtime.sendMessage({ what: 'setDefaultFilteringMode', level: 3 }))()`
    )

    if (completeLevel !== 3) {
      throw new Error('uBlock Origin Lite could not persist complete default filtering mode')
    }

    await popupWindow.loadURL(ublockState.popupUrl)
    const persistedLevel = await popupWindow.webContents.executeJavaScript(
      `(async () => chrome.runtime.sendMessage({ what: 'getDefaultFilteringMode' }))()`
    )

    if (persistedLevel !== 3) {
      throw new Error('uBlock Origin Lite default filtering mode did not persist after popup reload')
    }

    await popupWindow.loadURL(ublockState.dashboardUrl)
    const matchResult = await popupWindow.webContents.executeJavaScript(
      `(async () => chrome.declarativeNetRequest.testMatchOutcome({
        url: 'http://000491b06a.com/blocked.png',
        initiator: 'http://localhost/',
        method: 'get',
        type: 'image'
      }))()`
    )
    const matchedRules = (matchResult as { matchedRules?: unknown[] })?.matchedRules

    if (!Array.isArray(matchedRules) || matchedRules.length === 0) {
      throw new Error('the known blocked URL did not match a uBlock ruleset')
    }
  } finally {
    if (previewGuestId !== null) {
      runtime.unregisterGuest(previewGuestId)
    }
    if (ownerServer) {
      await new Promise<void>(resolve => ownerServer!.close(() => resolve()))
    }
    if (requestBlockerFixture) {
      fs.rmSync(requestBlockerFixture, { force: true, recursive: true })
    }
    if (ownerWindow && !ownerWindow.isDestroyed()) {
      ownerWindow.destroy()
    }

    const popupClosed =
      popupWindow && !popupWindow.isDestroyed()
        ? new Promise<void>(resolve => popupWindow!.once('closed', () => resolve()))
        : Promise.resolve()
    await runtime.setEnabled(false).catch(() => undefined)
    await runtime.dispose()
    await popupClosed

    if (popupWindow && !popupWindow.isDestroyed()) {
      cleanupError = new Error('uBlock Origin Lite popup was not cleaned up')
    }
  }

  if (cleanupError) {
    throw cleanupError
  }
}
