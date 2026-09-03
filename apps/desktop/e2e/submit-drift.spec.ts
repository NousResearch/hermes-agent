/**
 * Regression coverage for #69578: harmless route-token churn during a send
 * must not make the desktop silently drop the prompt before prompt.submit.
 */

import { test, expect } from './test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'

const PROMPT = 'E2E route token drift must still submit this prompt.'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture!, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('submits while same-chat search tokens churn during new-session creation', async ({}, testInfo) => {
  const { page, mock } = fixture!
  const composer = page.locator('[contenteditable="true"]').first()

  await composer.click()
  await composer.type(PROMPT, { delay: 10 })

  // The submit pipeline snapshots the route synchronously, then awaits session
  // creation. Keep changing only the query string of whichever chat route is
  // current. Before #69578, comparing the raw route token treated this as a
  // user chat switch and aborted before prompt.submit.
  await page.evaluate(() => {
    let revision = 0
    const interval = window.setInterval(() => {
      const pathname = window.location.hash.slice(1).split(/[?#]/, 1)[0] || '/new'
      window.location.hash = `${pathname}?e2e-route-churn=${revision++}`
    }, 1)

    ;(window as typeof window & { __e2eStopRouteChurn?: () => void }).__e2eStopRouteChurn = () => {
      window.clearInterval(interval)
    }
  })

  try {
    await page.keyboard.press('Enter')

    await expect
      .poll(() => mock.receivedPrompts.includes(PROMPT), { timeout: 60_000 })
      .toBe(true)

    await page.waitForFunction(
      prompt => document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent?.includes(prompt) ?? false,
      PROMPT,
      { timeout: 15_000 },
    )
    await page.waitForFunction(
      () => document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent?.includes('mock inference server') ?? false,
      undefined,
      { timeout: 60_000 },
    )
  } finally {
    await page.evaluate(() => {
      ;(window as typeof window & { __e2eStopRouteChurn?: () => void }).__e2eStopRouteChurn?.()
    })
  }

  await page.screenshot({ path: testInfo.outputPath('same-chat-route-churn-submitted.png') })
})