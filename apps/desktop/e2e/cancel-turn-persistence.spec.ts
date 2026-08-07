/**
 * Regression coverage for Stop while an assistant reply is streaming.
 *
 * Desktop keeps the visible partial reply optimistically when Stop is clicked.
 * That same partial reply must survive the actual interrupted-turn persistence
 * path: release the blocked provider stream, restart Desktop, then hydrate the
 * saved session again. Without persistence, the restart drops the partial
 * assistant work even though it appeared correct at the instant of cancellation.
 */

import { type TestInfo } from '@playwright/test'

import { expect, test, type Page } from './test'

import {
  buildAppEnv,
  launchDesktop,
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { CORRECTION_SWITCH_TRIGGER, MOCK_REPLY } from './mock-server'

const CANCELLED_PROMPT = `${CORRECTION_SWITCH_TRIGGER}: stop must retain prior streamed work.`
const STREAMED_WORK = 'Checking the long-running task before I continue.'
const INFERENCE_SESSION_PREFIX = 'E2E_CANCELLED_INFERENCE'
const INFERENCE_PROMPT = `${INFERENCE_SESSION_PREFIX}: keep only the partial reply after Stop.`
const FIRST_INFERENCE_TOKEN = 'Hello'

async function send(page: Page, text: string): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 15_000 })
  await composer.click()
  await composer.type(text, { delay: 5 })
  await page.keyboard.press('Enter')
}

async function waitForTranscriptText(page: Page, text: string, timeout = 30_000): Promise<void> {
  await page.waitForFunction(
    expected => (document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected),
    text,
    { timeout },
  )
}

test('Stop preserves streamed assistant text after Desktop restarts and rehydrates the session', async ({}, testInfo: TestInfo) => {
  const fixture: MockBackendFixture = await setupMockBackend()

  try {
    const { sandbox } = fixture
    let { page } = fixture
    await waitForAppReady(fixture, 120_000)

    await send(page, CANCELLED_PROMPT)
    // This trigger streams real assistant commentary then starts a foreground
    // tool which remains active for five seconds. Stop during that tool phase:
    // it exercises both the sealed interim message and the interrupted tool
    // history cleanup, rather than only a one-token inference cancellation.
    await waitForTranscriptText(page, STREAMED_WORK)

    const stop = page.locator('[data-slot="composer-root"] button[aria-label="Stop"]')
    await expect(stop).toBeVisible()
    await stop.click()
    await waitForTranscriptText(page, STREAMED_WORK)
    await page.screenshot({ path: testInfo.outputPath('cancelled-turn-before-restart.png') })

    // Let the cooperative tool cancellation settle so the gateway persists its
    // authoritative interrupted history before we restart Desktop.
    await expect(stop).toHaveCount(0)

    await fixture.app.close()
    const relaunched = await launchDesktop(buildAppEnv(sandbox))
    fixture.app = relaunched.app
    page = relaunched.page
    fixture.page = page
    await waitForAppReady(fixture, 120_000)

    // Session titles are preview-truncated in the persisted sidebar. Match the
    // stable trigger prefix rather than assuming the full prompt is retained as
    // the title, then activate it to exercise the real resume/hydration path.
    const sessionRow = page.locator('[data-slot="sidebar"] button').filter({ hasText: CORRECTION_SWITCH_TRIGGER }).first()
    await sessionRow.waitFor({ state: 'visible', timeout: 30_000 })
    await sessionRow.click()
    await waitForTranscriptText(page, CANCELLED_PROMPT)

    const transcript = page.locator('[data-slot="aui_thread-viewport"]')
    await expect(transcript).toContainText(CANCELLED_PROMPT)
    await expect(transcript).toContainText(STREAMED_WORK)
    await expect(transcript).toContainText('Ran sleep')
    await page.screenshot({ path: testInfo.outputPath('cancelled-turn-after-restart.png') })
  } finally {
    await fixture.cleanup()
  }
})

test('Stop preserves a partial inference reply after Desktop restarts and rehydrates the session', async ({}, testInfo: TestInfo) => {
  const fixture: MockBackendFixture = await setupMockBackend({
    mockServer: { holdFirstStreamForPrompt: INFERENCE_SESSION_PREFIX },
  })

  try {
    const { mock, sandbox } = fixture
    let { page } = fixture
    await waitForAppReady(fixture, 120_000)

    await send(page, INFERENCE_PROMPT)
    await mock.waitForHeldStream()
    await waitForTranscriptText(page, FIRST_INFERENCE_TOKEN)

    const stop = page.locator('[data-slot="composer-root"] button[aria-label="Stop"]')
    await expect(stop).toBeVisible()
    await stop.click()
    await waitForTranscriptText(page, FIRST_INFERENCE_TOKEN)
    await page.screenshot({ path: testInfo.outputPath('cancelled-inference-before-restart.png') })

    // The provider completes after Stop, so this proves Desktop does not merely
    // keep a live optimistic buffer — it must preserve the interrupted partial
    // through the backend's terminal event and subsequent persistence.
    mock.releaseHeldStream()
    await expect(stop).toHaveCount(0)

    await fixture.app.close()
    const relaunched = await launchDesktop(buildAppEnv(sandbox))
    fixture.app = relaunched.app
    page = relaunched.page
    fixture.page = page
    await waitForAppReady(fixture, 120_000)

    const sessionRow = page.locator('[data-slot="sidebar"] button').filter({ hasText: INFERENCE_SESSION_PREFIX }).first()
    await sessionRow.waitFor({ state: 'visible', timeout: 30_000 })
    await sessionRow.click()

    const transcript = page.locator('[data-slot="aui_thread-viewport"]')
    await expect(transcript).toContainText(INFERENCE_PROMPT)
    await expect(transcript).toContainText(FIRST_INFERENCE_TOKEN)
    await expect(transcript).not.toContainText(MOCK_REPLY)
    await page.screenshot({ path: testInfo.outputPath('cancelled-inference-after-restart.png') })
  } finally {
    await fixture.cleanup()
  }
})
