/**
 * Rendered proof that a live single clarification remains singular and
 * answerable while a warm session activation is waiting on REST hydration.
 */

import type { Agent, ClientRequest, RequestOptions } from 'node:http'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { BLOCKING_CLARIFY_QUESTION, BLOCKING_CLARIFY_TRIGGER, MOCK_REPLY } from './mock-server'
import { expect, type Page, test } from './test'

const OTHER_SESSION_PROMPT = 'E2E ordinary session for warm clarification activation.'
const CLARIFY_SESSION_PROMPT = `${BLOCKING_CLARIFY_TRIGGER}: keep the clarification answerable during warm activation.`
const SURFACE = '[data-composer-target]:not([data-pane-hidden] [data-composer-target])'
const REST_GATE_GLOBAL = '__HERMES_E2E_CLARIFY_REST_GATE__'

interface MainProcessRestGate {
  held: boolean
  path: string | null
  released: boolean
  restored: boolean
  release: () => void
  readHeldResponse: () => Promise<string>
  restore: () => void
}

interface HeldRequest {
  agent: PatchableAgent
  request: ClientRequest
  options: RequestOptions
}

type PatchableAgent = Agent & {
  addRequest: (request: ClientRequest, options: RequestOptions) => void
}

type GateGlobal = typeof globalThis & {
  [REST_GATE_GLOBAL]?: MainProcessRestGate
}

function activeSurface(page: Page) {
  return page.locator(SURFACE).last()
}

async function send(page: Page, text: string): Promise<void> {
  const composer = activeSurface(page).locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 15_000 })
  await composer.click()
  await composer.type(text, { delay: 5 })
  await page.keyboard.press('Enter')
}

async function waitForTranscriptText(page: Page, text: string): Promise<void> {
  await page.waitForFunction(
    ([expected, surfaceSelector]: [string, string]) => {
      const surfaces = document.querySelectorAll(surfaceSelector)
      const active = surfaces[surfaces.length - 1]

      return (active?.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected)
    },
    [text, SURFACE] as [string, string],
    { timeout: 60_000 }
  )
}

async function openFreshDraft(page: Page, priorSessionText: string): Promise<void> {
  await page.locator('[data-slot="sidebar"] button[aria-label="New session"]').first().click()
  await page.waitForFunction(
    ([priorText, surfaceSelector]: [string, string]) => {
      const surfaces = document.querySelectorAll(surfaceSelector)
      const active = surfaces[surfaces.length - 1]
      const transcript = active?.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? ''

      return surfaces.length > 0 && !transcript.includes(priorText)
    },
    [priorSessionText, SURFACE] as [string, string],
    { timeout: 15_000 }
  )
}

async function openSidebarSession(page: Page, sidebarText: string): Promise<void> {
  const row = page.locator('[data-slot="sidebar"] button').filter({ hasText: sidebarText }).first()
  await row.waitFor({ state: 'visible', timeout: 30_000 })
  await row.click()
}

async function installNextSessionMessagesGetGate(fixture: MockBackendFixture, storedSessionId: string): Promise<void> {
  await fixture.app.evaluate(
    (_electron, args) => {
      const { globalName, targetPath } = args
      const http = process.getBuiltinModule('node:http')
      const prototype = http.Agent.prototype as PatchableAgent
      const original = prototype.addRequest
      let heldRequest: HeldRequest | null = null
      let timeout: ReturnType<typeof setTimeout> | null = null

      const state: MainProcessRestGate = {
        held: false,
        path: null,
        released: false,
        restored: false,
        release: () => {
          if (state.released) {
            return
          }

          state.released = true
          const pending = heldRequest
          heldRequest = null

          if (pending) {
            original.call(pending.agent, pending.request, pending.options)
          }

          state.restore()
        },
        readHeldResponse: () =>
          new Promise<string>((resolve, reject) => {
            if (!heldRequest) {
              resolve('')

              return
            }

            const probe = http.request({ ...heldRequest.options, method: 'GET' }, response => {
              response.setEncoding('utf8')
              let body = ''

              response.on('data', chunk => {
                body += chunk
              })
              response.on('end', () => resolve(body))
            })

            probe.on('error', reject)
            probe.end()
          }),
        restore: () => {
          if (prototype.addRequest === patchedAddRequest) {
            prototype.addRequest = original
          }

          state.restored = true

          if (timeout) {
            clearTimeout(timeout)
          }

          timeout = null
        }
      }

      function patchedAddRequest(this: PatchableAgent, request: ClientRequest, options: RequestOptions): void {
        const method = String(options.method ?? request.method ?? 'GET').toUpperCase()
        const requestPath = String(options.path ?? request.path ?? '')

        const isTarget = method === 'GET' && requestPath.includes(targetPath)

        if (!state.held && isTarget) {
          state.held = true
          state.path = requestPath
          heldRequest = { agent: this, request, options }

          return
        }

        original.call(this, request, options)
      }

      ;(globalThis as GateGlobal)[globalName as typeof REST_GATE_GLOBAL] = state
      prototype.addRequest = patchedAddRequest
      timeout = setTimeout(() => state.release(), 30_000)
    },
    {
      globalName: REST_GATE_GLOBAL,
      targetPath: `/api/sessions/${encodeURIComponent(storedSessionId)}/messages`
    }
  )
}

async function gateSnapshot(
  fixture: MockBackendFixture
): Promise<Pick<MainProcessRestGate, 'held' | 'path' | 'released' | 'restored'> | null> {
  return fixture.app.evaluate((_electron, globalName) => {
    const state = (globalThis as GateGlobal)[globalName as typeof REST_GATE_GLOBAL]

    if (!state) {
      return null
    }

    return {
      held: state.held,
      path: state.path,
      released: state.released,
      restored: state.restored
    }
  }, REST_GATE_GLOBAL)
}

async function releaseGate(fixture: MockBackendFixture): Promise<void> {
  await fixture.app.evaluate((_electron, globalName) => {
    ;(globalThis as GateGlobal)[globalName as typeof REST_GATE_GLOBAL]?.release()
  }, REST_GATE_GLOBAL)
}

async function heldResponse(fixture: MockBackendFixture): Promise<string> {
  return fixture.app.evaluate((_electron, globalName) => {
    const state = (globalThis as GateGlobal)[globalName as typeof REST_GATE_GLOBAL]

    return state?.readHeldResponse() ?? ''
  }, REST_GATE_GLOBAL)
}

async function releaseAndRestoreGate(fixture: MockBackendFixture): Promise<void> {
  await fixture.app
    .evaluate((_electron, globalName) => {
      const state = (globalThis as GateGlobal)[globalName as typeof REST_GATE_GLOBAL]
      state?.release()
      state?.restore()
    }, REST_GATE_GLOBAL)
    .catch(() => undefined)
}

async function expectSingleAnswerableClarification(page: Page): Promise<void> {
  const surface = activeSurface(page)
  const form = surface.locator('form').filter({ hasText: BLOCKING_CLARIFY_QUESTION })
  const yes = form.locator('button').filter({ hasText: 'Yes' })
  const no = form.locator('button').filter({ hasText: 'No' })

  await expect(form).toHaveCount(1)
  await expect(form.getByText(BLOCKING_CLARIFY_QUESTION, { exact: true })).toHaveCount(1)
  await expect(yes).toHaveCount(1)
  await expect(no).toHaveCount(1)
  await expect(yes).toBeEnabled()
  await expect(no).toBeEnabled()
  await expect(surface.getByText(OTHER_SESSION_PROMPT, { exact: true })).toHaveCount(0)
  await expect(surface.getByText(MOCK_REPLY, { exact: true })).toHaveCount(0)
}

test.describe('warm clarification activation', () => {
  test.setTimeout(180_000)

  let fixture: MockBackendFixture | null = null

  test.beforeEach(async () => {
    fixture = await setupMockBackend()
    await waitForAppReady(fixture, 120_000)
  })

  test.afterEach(async () => {
    if (fixture) {
      await releaseAndRestoreGate(fixture)
    }

    await fixture?.cleanup()
    fixture = null
  })

  test('keeps one answerable clarification rendered while activation REST is held', async ({
    page: _unusedPage
  }, testInfo) => {
    const current = fixture!
    const { page } = current

    await send(page, OTHER_SESSION_PROMPT)
    await waitForTranscriptText(page, MOCK_REPLY)
    await openFreshDraft(page, OTHER_SESSION_PROMPT)

    await send(page, CLARIFY_SESSION_PROMPT)
    await activeSurface(page)
      .getByText(BLOCKING_CLARIFY_QUESTION, { exact: true })
      .waitFor({ state: 'visible', timeout: 60_000 })

    const activeClarifyTab = page.locator('[role="tab"][aria-selected="true"][data-tree-tab^="session-tile:"]').last()
    await expect(activeClarifyTab).toBeVisible()
    const clarifyPaneId = await activeClarifyTab.getAttribute('data-tree-tab')
    expect(clarifyPaneId).toMatch(/^session-tile:.+/)
    const clarifyStoredSessionId = clarifyPaneId!.slice('session-tile:'.length)

    await activeClarifyTab.hover()
    await activeClarifyTab.getByRole('button', { name: /close/i }).click()
    const closeDialog = page.getByRole('dialog', { name: /close running tab/i })
    await expect(closeDialog).toBeVisible()
    await closeDialog.getByRole('button', { name: 'Close tab', exact: true }).click()
    await waitForTranscriptText(page, OTHER_SESSION_PROMPT)
    await expect(activeSurface(page).getByText(BLOCKING_CLARIFY_QUESTION, { exact: true })).toHaveCount(0)

    await installNextSessionMessagesGetGate(current, clarifyStoredSessionId)

    try {
      await openSidebarSession(page, 'E2E_BLOCKING_CLARIFY')
      await expect
        .poll(() => gateSnapshot(current), { timeout: 15_000 })
        .toMatchObject({
          held: true,
          released: false,
          restored: false
        })
      const held = await gateSnapshot(current)
      expect(held?.path).toContain('/api/sessions/')
      expect(held?.path).toContain(encodeURIComponent(clarifyStoredSessionId))
      expect(held?.path).toContain('/messages')
      const heldBody = await heldResponse(current)
      expect(heldBody).toContain(CLARIFY_SESSION_PROMPT)
      expect(heldBody).toContain(BLOCKING_CLARIFY_QUESTION)

      await expectSingleAnswerableClarification(page)
      await page.screenshot({ path: testInfo.outputPath('clarification-while-rest-held.png') })

      await releaseGate(current)
      await expect
        .poll(() => gateSnapshot(current))
        .toMatchObject({
          held: true,
          released: true,
          restored: true
        })
      await expectSingleAnswerableClarification(page)
      await page.screenshot({ path: testInfo.outputPath('clarification-after-rest-release.png') })
    } finally {
      await releaseAndRestoreGate(current)
    }
  })
})
