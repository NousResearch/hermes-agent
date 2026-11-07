import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { MOCK_REPLY, receivedUserTexts, startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, type MockBackendFixture, waitForAppReady } from './fixtures'
import { expect, test } from './test'

interface ReconnectProbe {
  blocked: boolean
  sockets: Set<WebSocket>
  requests: Map<string, { method: string; sessionId?: string }>
  resumed: string[]
  rejectedApprovals: string[]
  submitted: string[]
}

declare global {
  interface Window {
    __approvalReconnectProbe: ReconnectProbe
  }
}

let fixture: MockBackendFixture | undefined

test.afterEach(async () => {
  await fixture?.cleanup()
})

test('reconnects after runtime reclamation without querying approvals on the retired runtime', async () => {
  test.setTimeout(180_000)
  const mock = await startMockServer()
  const sandbox = createSandbox('approval-reconnect')

  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome, 'e2e-mock-key', mock.url)
  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  fixture = {
    app, page, mock, mockUrl: mock.url, sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }
  await waitForAppReady(fixture, 90_000)

  await page.evaluate(() => {
    const NativeWebSocket = window.WebSocket
    const send = NativeWebSocket.prototype.send

    const probe: ReconnectProbe = {
      blocked: false, sockets: new Set(), requests: new Map(), resumed: [], rejectedApprovals: [], submitted: []
    }

    window.__approvalReconnectProbe = probe

    NativeWebSocket.prototype.send = function (data) {
      if (!probe.sockets.has(this)) {
        probe.sockets.add(this)
        this.addEventListener('message', event => {
          const frame = JSON.parse(String(event.data))
          const request = probe.requests.get(String(frame.id))

          if (request?.method === 'approval.pending' && frame.error?.code === 4001) {
            probe.rejectedApprovals.push(request.sessionId ?? '')
          }

          if (request?.method === 'session.resume' && frame.result?.session_id) {
            probe.resumed.push(frame.result.session_id)
          }
        })
      }

      const frame = JSON.parse(String(data))

      if (frame.method) {
        probe.requests.set(String(frame.id), { method: frame.method, sessionId: frame.params?.session_id })

        if (frame.method === 'prompt.submit') {
          probe.submitted.push(frame.params.session_id)
        }
      }

      return send.call(this, data)
    }

    window.WebSocket = class extends NativeWebSocket {
      constructor(url: string | URL, protocols?: string | string[]) {
        super(url, protocols)

        if (probe.blocked) {
          queueMicrotask(() => this.close())
        }
      }
    }
  })

  const composer = page.locator('[contenteditable="true"]').first()
  await composer.fill('E2E_APPROVAL_RECONNECT_BEFORE')
  await composer.press('Enter')
  await expect(page.locator('[data-slot="aui_thread-viewport"]')).toContainText(MOCK_REPLY, { timeout: 60_000 })
  const oldRuntime = await page.evaluate(() => window.__approvalReconnectProbe.submitted.at(-1))

  expect(oldRuntime).toBeTruthy()
  await page.evaluate(() => {
    const probe = window.__approvalReconnectProbe

    probe.blocked = true
    probe.resumed.length = 0

    for (const socket of probe.sockets) {
      socket.close()
    }
  })
  // Exceed the real backend's default 20-second orphan grace.
  await page.waitForTimeout(25_000)
  await page.evaluate(() => { window.__approvalReconnectProbe.blocked = false })
  await expect.poll(
    () => page.evaluate(() => window.__approvalReconnectProbe.resumed.at(-1)),
    { timeout: 45_000 }
  ).toEqual(expect.any(String))
  expect(await page.evaluate(() => window.__approvalReconnectProbe.resumed.at(-1))).not.toBe(oldRuntime)

  await composer.fill('E2E_APPROVAL_RECONNECT_AFTER')
  await composer.press('Enter')
  await expect.poll(() => receivedUserTexts().includes('E2E_APPROVAL_RECONNECT_AFTER'), { timeout: 30_000 }).toBe(true)
  await expect(page.locator('[data-slot="aui_thread-viewport"]')).toContainText('E2E_APPROVAL_RECONNECT_BEFORE')
  expect(await page.evaluate(() => window.__approvalReconnectProbe.rejectedApprovals)).toEqual([])
})
