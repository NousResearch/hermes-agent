/**
 * E2E regression for the gateway file download bridge (upstream #90580
 * family). The main-process handler behind `hermes:saveGatewayFile` must
 * ALWAYS reply: any failure resolves with a legible {saved: false, error}
 * carrying the real cause (404 from the backend, missing path, timeout) —
 * never Electron's opaque "reply was never sent", which is what an
 * uncloneable rejection or an unresolved handler produced.
 *
 * Deterministic IPC-level repro inside the running app: the bridge is called
 * with a path that cannot exist on the backend, so the real `hermes serve`
 * answers 404 and the normalized outcome must reach the renderer.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */
import { expect, test } from './test'
import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture!, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test.describe('gateway file download bridge always replies', () => {
  test('a missing backend file resolves {saved:false} with a legible 404 cause', async () => {
    const outcome = await fixture!.page.evaluate(async () => {
      try {
        return await window.hermesDesktop.saveGatewayFile({
          path: '/definitively/missing/informe final ñandú.md',
          suggestedName: 'informe final ñandú.md'
        })
      } catch (error) {
        // The pre-fix handler rejected non-cloneably; capture that shape so the
        // assertion failure names the regression instead of a generic timeout.
        return { canceled: false, error: `REJECTED: ${(error as Error)?.message ?? String(error)}`, saved: false }
      }
    })

    expect(outcome.saved).toBe(false)
    expect(outcome.canceled).toBeFalsy()
    const reason = String(outcome.error ?? '')
    expect(reason.toLowerCase()).not.toContain('reply was never sent')
    expect(reason.toLowerCase()).toMatch(/404|not found|no file data|missing/)
  })

  test('a blank path resolves {saved:false, error:"Missing gateway file path"}', async () => {
    const outcome = await fixture!.page.evaluate(async () =>
      await window.hermesDesktop.saveGatewayFile({ path: '   ', suggestedName: 'x.md' }))

    expect(outcome.saved).toBe(false)
    expect(outcome.canceled).toBeFalsy()
    expect(outcome.error).toBe('Missing gateway file path')
  })
})
