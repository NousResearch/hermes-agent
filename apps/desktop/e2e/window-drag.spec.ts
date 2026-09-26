import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

let fixture: MockBackendFixture

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
})

test('the window top edge stays draggable after typing into a session tab', async () => {
  const testInfo = test.info()
  const { page } = fixture
  const composer = page.locator('[data-slot="composer-rich-input"]:visible').first()
  await composer.click()
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+t' : 'Control+t')
  const tabs = page.locator('[data-zone-tabstrip="grp-main"] [data-tree-tab]')
  await expect(tabs).toHaveCount(2)
  await composer.click()
  await composer.pressSequentially('Keep this draft while moving the window')

  const edge = await tabs.last().evaluate(tab => {
    const strip = tab.closest('[data-zone-tabstrip]')!
    const header = strip.getBoundingClientRect()
    const rect = tab.getBoundingClientRect()
    const point = { x: rect.left + rect.width / 2, y: header.top + 4 }

    const blockers = [...document.querySelectorAll('*')].filter(element => {
      const style = getComputedStyle(element)
      const bounds = element.getBoundingClientRect()

      return (
        style.getPropertyValue('-webkit-app-region') === 'no-drag' &&
        style.visibility === 'visible' &&
        bounds.left <= point.x &&
        bounds.right > point.x &&
        bounds.top <= point.y &&
        bounds.bottom > point.y
      )
    })

    return {
      dragRegion: getComputedStyle(strip).getPropertyValue('-webkit-app-region'),
      gap: rect.top - header.top,
      blockers: blockers.map(element => element.getAttribute('data-tree-tab') ?? element.tagName)
    }
  })

  await page.screenshot({ path: testInfo.outputPath('typing-in-session-tab.png') })
  expect(edge.dragRegion).toBe('drag')
  expect(edge.gap).toBeGreaterThan(4)
  expect(edge.blockers).toEqual([])

  // The grip must not take the tab's normal click target or steal the draft.
  await tabs.first().click()
  await expect(tabs.first()).toHaveAttribute('aria-selected', 'true')
  await tabs.last().click()
  await expect(tabs.last()).toHaveAttribute('aria-selected', 'true')
  await expect(composer).toHaveText('Keep this draft while moving the window')
})
