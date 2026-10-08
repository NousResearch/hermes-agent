/**
 * C2 core: steering and queueing with NeMo Relay managed execution on.
 *
 * Opting into shared metrics routes every provider stream through Relay's
 * managed pipeline. Relay pulls the next provider chunk before it hands over
 * the current one, so a blocking provider read withheld each chunk until the
 * provider sent the next: the reply looked stopped while the timer kept going,
 * and a steer aborted text the user never saw, stacking user bubbles with no
 * output between them. These steps hold the provider mid-stream and assert the
 * text is already on screen, then steer and queue on top of it.
 */

import { expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  send,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler, type OracleTarget } from './oracle'
import { gate, startScriptedProvider } from './provider'

const RELAY_MANAGED_EXECUTION = `telemetry:
  shared_metrics:
    enabled: true
`

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`

const words = (marker: string, ...rest: string[]) => [
  `${marker} `,
  ...rest.map((w, i) => (i === rest.length - 1 ? w : `${w} `))
]

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

function stopButton(page: Page) {
  return page.locator('[data-slot="composer-root"] button[aria-label="Stop"]')
}

test('steer and queue stay live under Relay managed execution', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('relay-steer')
  writeProviderHome(sandbox.hermesHome, provider.url, RELAY_MANAGED_EXECUTION)
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)

  const finished = (marker: string) =>
    expect
      .poll(() => provider.completions.some(c => c.marker === marker && c.finished), {
        timeout: 120_000,
        message: `provider finished ${marker}`
      })
      .toBe(true)

  const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }

  try {
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)

    await test.step('steer lands below the partial reply the user already saw', async () => {
      const hold = gate()
      provider.script(U(1), [{ text: words(A(1), 'slow', 'first', 'reply'), holdAfterFirstChunk: hold }])
      provider.script(U(2), [{ text: words(A(2), 'steered', 'reply') }])
      await send(page, `${U(1)} slow one`, 'Enter', ws)
      await provider.streamStarted(U(1))
      // The provider is paused after its first chunk; that chunk must already be painted.
      await expect(viewport(page)).toContainText(A(1), { timeout: 10_000 })
      await send(page, `${U(2)} change course`, 'Enter', ws)
      await finished(U(2))
      hold.open()
      session.sessionId = await currentSessionId(page)
      session.expectUserMarkers.push(U(1), U(2))
      await assertTranscriptOracle(page, ws, provider, session, 'relay steer')
    })

    await test.step('a follow-up queued under a steered reply drains and the turn settles', async () => {
      const hold = gate()
      const steerHold = gate()
      provider.script(U(3), [{ text: words(A(3), 'long', 'running', 'reply'), holdAfterFirstChunk: hold }])
      provider.script(U(4), [{ text: words(A(4), 'steered', 'again'), holdAfterFirstChunk: steerHold }])
      provider.script(U(5), [{ text: words(A(5), 'queued', 'reply') }])
      await send(page, `${U(3)} long one`, 'Enter', ws)
      await provider.streamStarted(U(3))
      await expect(viewport(page)).toContainText(A(3), { timeout: 10_000 })
      await send(page, `${U(4)} steer it`, 'Enter', ws)
      await provider.streamStarted(U(4))
      await expect(viewport(page)).toContainText(A(4), { timeout: 10_000 })
      await send(page, `${U(5)} after that`, 'Control+Enter', ws)
      hold.open()
      steerHold.open()
      await finished(U(5))
      await expect(stopButton(page)).toHaveCount(0, { timeout: 30_000 })
      session.expectUserMarkers.push(U(3), U(4), U(5))
      await assertTranscriptOracle(page, ws, provider, session, 'relay queued follow-up')
    })
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
