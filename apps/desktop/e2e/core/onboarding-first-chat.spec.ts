/**
 * C2 core: onboarding completion, then the first chat.
 *
 * Fresh sandbox, NO provider configured: the real first-run onboarding shows.
 * The user picks the local / custom endpoint path and enters the fake
 * provider's URL; the real backend probes it (/v1/models), persists the
 * assignment, and the overlay closes. Then the first chat must obey the same
 * transcript oracle as everything else. A smoke test for onboarding followed
 * by a first chat; the #120005 double-paint needs a non-default profile and is
 * guarded by transcript-integrity.spec.ts.
 *
 * Invariants: the onboarding result is what the backend persisted and what
 * the model is actually called with (config base_url == entered URL; the
 * first completion carries the endpoint's advertised model); one live socket
 * to the backend after onboarding; every message exactly once, DOM ==
 * persisted, streamed == provider, at every sampled frame.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  send,
  waitForInteractive
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler } from './oracle'
import { startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`

test('onboarding (custom endpoint) then first chat renders exactly once', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('onboard')
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)

  try {
    await installDuplicateSampler(page)

    await test.step('first-run onboarding: custom endpoint', async () => {
      // The onboarding surface masks the whole app until it finishes.
      const openKeyForm = page
        .locator('[data-glass-opaque]')
        .filter({ visible: true })
        .getByRole('button', { name: /api key/i })

      await expect(openKeyForm).toHaveCount(1, { timeout: 180_000 })
      await openKeyForm.click()
      await page
        .locator('[data-glass-opaque]')
        .filter({ visible: true })
        .getByRole('button', { name: /custom endpoint/i })
        .click()

      // The endpoint URL is the form's only plain-text input (API keys are password inputs).
      const form = page
        .locator('[data-glass-opaque]')
        .filter({ visible: true })
        .filter({ has: page.locator('input[type="text"]') })

      await expect(form).toHaveCount(1)
      await form.locator('input[type="text"]').fill(provider.url)
      await form.getByRole('button', { name: /connect/i }).click()
      // The overlay unmounts only after the backend probed the endpoint, saved
      // the assignment and confirmed the runtime is ready; waitForInteractive
      // fails while any full-viewport fixed layer still covers the composer.
      await waitForInteractive(app, page)
    })

    await test.step('the persisted assignment is what the user entered', async () => {
      const config = fs.readFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'utf8')
      expect(config).toMatch(/provider:\s*['"]?custom/)
      expect(config).toContain(`127.0.0.1:${new URL(provider.url).port}`)
    })

    await test.step('first chat after onboarding renders exactly once', async () => {
      provider.script(U(1), [
        { reasoning: ['R1-', nonce, ' thinking'], text: [`${A(1)} `, 'first ', 'reply ', 'after ', 'setup'] }
      ])
      await send(page, `${U(1)} hello`, 'Enter', ws)
      await expect
        .poll(() => provider.completions.some(c => c.marker === U(1) && c.finished), {
          timeout: 120_000,
          message: 'the first chat after onboarding reached the configured endpoint'
        })
        .toBe(true)
      const first = provider.completions.find(c => c.marker === U(1))
      // The fake endpoint advertises exactly one model at /v1/models.
      expect(first?.body?.model, 'the model is called with the endpoint-advertised model').toBe('mock-model')
      await expect.poll(() => currentSessionId(page)).not.toBe('')
      const sessionId = await currentSessionId(page)
      await assertTranscriptOracle(
        page,
        ws,
        provider,
        { sessionId, expectUserMarkers: [U(1)] },
        'first chat after onboarding'
      )
      await expect
        .poll(() => ws.sockets.filter(s => !s.closed).length, { timeout: 30_000, message: 'live backend sockets' })
        .toBe(1)
    })

    await test.step('second turn and a reload keep it exactly once', async () => {
      provider.script(U(2), [{ text: [`${A(2)} `, 'second ', 'reply'] }])
      await send(page, `${U(2)} again`, 'Enter', ws)
      await expect
        .poll(() => provider.completions.some(c => c.marker === U(2) && c.finished), { timeout: 120_000 })
        .toBe(true)
      const sessionId = await currentSessionId(page)
      await assertTranscriptOracle(page, ws, provider, { sessionId, expectUserMarkers: [U(1), U(2)] }, 'second turn')
      await page.reload()
      await waitForInteractive(app, page)
      await installDuplicateSampler(page)
      await expect.poll(() => currentSessionId(page)).toBe(sessionId)
      await assertTranscriptOracle(
        page,
        ws,
        provider,
        { sessionId, expectUserMarkers: [U(1), U(2)] },
        'reload after onboarding'
      )
    })
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
