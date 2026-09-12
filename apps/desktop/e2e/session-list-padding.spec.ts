/**
 * Session-list row geometry checks for padding and descender visibility.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, test } from './test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

const LONG_TITLE =
  'gypq session title with enough additional words to verify that the session list keeps ellipsis while preserving descenders'

async function sendMessage(page: MockBackendFixture['page'], text: string): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 10_000 })
  await composer.click()
  await composer.type(text, { delay: 5 })
  await page.keyboard.press('Enter')
  await page.getByText('boot chain is working', { exact: false }).first().waitFor({ state: 'visible', timeout: 60_000 })
}

async function assertSessionRows(fixture: MockBackendFixture, density: 'compact' | 'comfortable'): Promise<void> {
  const { page } = fixture
  await page.evaluate(async value => {
    localStorage.setItem('hermes.desktop.sessionListDensity', value)
    await document.fonts.ready
  }, density)
  await page.reload()
  await waitForAppReady(fixture, 120_000)
  await page.evaluate(() => document.fonts.ready)

  const metrics = await page.locator('[data-slot="row-button"]').evaluateAll(rows =>
    rows
      .map(row => {
        const shell = row.parentElement
        const actions = shell?.querySelector<HTMLElement>('[data-row-actions]')
        const title = row.querySelector<HTMLElement>('.hover-marquee')
        const timestamp = actions?.querySelector<HTMLElement>('time')

        if (!shell || !actions || !title || !timestamp) return null

        const rowRect = shell.getBoundingClientRect()
        const titleRect = title.getBoundingClientRect()
        const timestampRect = timestamp.getBoundingClientRect()
        const style = getComputedStyle(row)
        const clippingAncestors: Array<{ top: number; bottom: number; left: number; right: number }> = []

        for (let ancestor = title.parentElement; ancestor; ancestor = ancestor.parentElement) {
          const overflow = getComputedStyle(ancestor).overflow
          if (overflow === 'hidden' || overflow === 'clip' || overflow === 'scroll' || overflow === 'auto') {
            const rect = ancestor.getBoundingClientRect()
            clippingAncestors.push({ top: rect.top, bottom: rect.bottom, left: rect.left, right: rect.right })
          }
          if (ancestor === shell) break
        }

        return {
          titleText: title.textContent,
          start: Number.parseFloat(style.paddingInlineStart),
          end: Number.parseFloat(style.paddingInlineEnd),
          timestampInset: rowRect.right - timestampRect.right,
          titleScrollHeight: title.scrollHeight,
          titleClientHeight: title.clientHeight,
          titleScrollWidth: title.scrollWidth,
          titleClientWidth: title.clientWidth,
          titleRect,
          clippingAncestors
        }
      })
      .filter(Boolean)
  )

  expect(metrics.length, `${density} should render session rows`).toBeGreaterThanOrEqual(2)
  expect(
    metrics.some(row => row?.titleText?.includes('gypq')),
    'a descender title should be present'
  ).toBe(true)

  for (const row of metrics) {
    expect(Math.abs(row!.start - row!.end), `${density} row padding should be symmetric`).toBeLessThanOrEqual(1)
    expect(row!.timestampInset + 1, `${density} timestamp should clear trailing padding`).toBeGreaterThanOrEqual(
      row!.end
    )
    expect(row!.titleScrollHeight, `${density} title descenders should fit`).toBeLessThanOrEqual(
      row!.titleClientHeight + 1
    )
    expect(row!.titleScrollWidth, `${density} long title should ellipsize`).toBeGreaterThan(row!.titleClientWidth)

    for (const clip of row!.clippingAncestors) {
      expect(row!.titleRect.top + 1).toBeGreaterThanOrEqual(clip.top)
      expect(row!.titleRect.bottom - 1).toBeLessThanOrEqual(clip.bottom)
      expect(row!.titleRect.left + 1).toBeGreaterThanOrEqual(clip.left)
      expect(row!.titleRect.right - 1).toBeLessThanOrEqual(clip.right)
    }
  }
}

test.describe('session list padding and descenders', () => {
  let fixture: MockBackendFixture

  test.beforeAll(async () => {
    fixture = await setupMockBackend({ mockServer: { holdFirstCompletionContaining: 'gypq' } })
    await waitForAppReady(fixture, 120_000)
    await sendMessage(fixture.page, 'seed session')
    await fixture.page.getByRole('button', { name: 'New session', exact: true }).click()
    const composer = fixture.page.locator('[contenteditable="true"]:visible').first()
    await composer.click()
    await composer.type(LONG_TITLE, { delay: 5 })
    await fixture.page.keyboard.press('Enter')
    await fixture.mock.waitForHeldCompletion()
    await fixture.page
      .locator('[data-slot="row-button"]')
      .filter({ hasText: 'gypq' })
      .waitFor({ state: 'visible', timeout: 30_000 })
  })

  test.afterAll(async () => {
    fixture?.mock.releaseHeldStream()
    await fixture?.cleanup()
  })

  test('keeps compact rows padded and unclipped', async () => {
    await assertSessionRows(fixture, 'compact')
  })

  test('keeps comfortable rows padded and unclipped', async () => {
    await assertSessionRows(fixture, 'comfortable')
  })
})
