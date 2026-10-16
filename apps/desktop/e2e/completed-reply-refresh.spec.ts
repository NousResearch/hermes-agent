import { expect, test } from '@playwright/test'

import { MOCK_REPLY } from '../../../tests-js/scripts/mock-server'

import { setupMockBackend, waitForAppReady } from './fixtures'

// Real Electron, backend, SQLite and stream. Only delivery of one real history
// request is delayed: admit while idle, read while the next turn is held, then
// deliver after completion. The mock provider intentionally repeats its answer.
test('a delayed history read cannot remove the latest completed reply', async () => {
  test.setTimeout(150_000)
  const nextPrompt = 'Complete the second request for the refresh regression'
  const fixture = await setupMockBackend({ mockServer: { holdFirstStreamForPrompt: nextPrompt } })
  const { app, page, mock } = fixture
  const events: string[] = []
  page.on('websocket', socket => {
    socket.on('framereceived', frame => {
      try {
        const message = JSON.parse(String(frame.payload))

        if (message.method === 'event') {
          events.push(message.params?.type)
        }
      } catch {
        /* Non-JSON transport frames carry no turn event. */
      }
    })
  })

  try {
    await waitForAppReady(fixture)

    const composer = page
      .locator('[data-slot="composer-root"] [contenteditable="true"]')
      .filter({ visible: true })
      .first()

    await composer.fill('First request for the refresh regression')
    await composer.press('Enter')
    const answers = page.locator('[data-slot="aui_assistant-message-content"]').filter({ hasText: MOCK_REPLY })
    await expect(answers).toHaveCount(1, { timeout: 60_000 })
    await expect.poll(() => events.filter(type => type === 'message.complete').length).toBeGreaterThan(0)
    const stored = await page.evaluate(() => decodeURIComponent(location.hash.slice(2).split('?')[0]))
    expect(stored).not.toBe('')

    // The private IPC handler table is used only by this test. Preserve the
    // original handler (including backend routing and actual SQLite reads).
    await app.evaluate(({ ipcMain }, sessionId) => {
      const handlers = (ipcMain as unknown as { _invokeHandlers: Map<string, (...args: any[]) => Promise<any>> })
        ._invokeHandlers

      const original = handlers.get('hermes:api')!

      const control = {
        admitted: false,
        sampled: false,
        releaseRead: () => {},
        releaseDelivery: () => {},
        snapshot: null as any
      }

      const readGate = new Promise<void>(resolve => {
        control.releaseRead = resolve
      })

      const deliveryGate = new Promise<void>(resolve => {
        control.releaseDelivery = resolve
      })

      ;(globalThis as any).__historyRead = control
      ipcMain.removeHandler('hermes:api')
      ipcMain.handle('hermes:api', async (event: unknown, request: { path: string }) => {
        if (!control.admitted && request.path.startsWith(`/api/sessions/${sessionId}/messages?`)) {
          control.admitted = true
          await readGate
          const snapshot = await original(event, request)
          control.snapshot = snapshot
          control.sampled = true
          await deliveryGate

          return snapshot
        }

        return original(event, request)
      })
    }, stored)

    // An actual backend metadata write produces the production change tick.
    await page.evaluate(async sessionId => {
      await (window as any).hermesDesktop.api({
        path: `/api/sessions/${sessionId}`,
        method: 'PATCH',
        body: { title: 'Completed reply regression' }
      })
    }, stored)
    await expect
      .poll(() => app.evaluate(() => (globalThis as any).__historyRead.admitted), { timeout: 20_000 })
      .toBe(true)
    await composer.fill(nextPrompt)
    await composer.press('Enter')
    await mock.waitForHeldStream()
    await app.evaluate(() => {
      ;(globalThis as any).__historyRead.releaseRead()
    })
    await expect.poll(() => app.evaluate(() => (globalThis as any).__historyRead.sampled)).toBe(true)
    const sampled = await app.evaluate(() => (globalThis as any).__historyRead.snapshot)
    expect(sampled.messages.some((message: any) => message.role === 'user' && message.content === nextPrompt)).toBe(
      true
    )
    expect(
      sampled.messages.filter((message: any) => message.role === 'assistant' && message.content === MOCK_REPLY)
    ).toHaveLength(1)
    const completeCount = events.filter(type => type === 'message.complete').length
    mock.releaseHeldStream()
    await expect
      .poll(() => events.filter(type => type === 'message.complete').length, { timeout: 60_000 })
      .toBeGreaterThan(completeCount)
    await expect(answers).toHaveCount(2)
    await page.screenshot({ path: test.info().outputPath('completed-before-history.png') })
    await app.evaluate(() => {
      ;(globalThis as any).__historyRead.releaseDelivery()
    })
    // A bounded quiet window checks that the answer stays painted after the
    // delayed IPC promise and the renderer effects have been processed.
    await page.waitForTimeout(1500)
    await expect(answers).toHaveCount(2)
    await expect(page.locator('[data-slot="aui_thread-viewport"]')).toContainText(nextPrompt)
    await page.screenshot({ path: test.info().outputPath('completed-after-history.png') })
    await test.info().attach('transport-receipt', {
      body: JSON.stringify(
        { stored, events, sampledRows: sampled.messages.length, visibleAnswers: await answers.count() },
        null,
        2
      ),
      contentType: 'application/json'
    })
  } finally {
    await app
      .evaluate(() => {
        ;(globalThis as any).__historyRead?.releaseRead()
        ;(globalThis as any).__historyRead?.releaseDelivery()
      })
      .catch(() => undefined)
    mock.releaseHeldStream()
    await fixture.cleanup()
  }
})
