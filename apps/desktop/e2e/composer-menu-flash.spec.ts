import { test, expect } from './test'
import { setupMockBackend, waitForAppReady } from './fixtures'

test('goal prose never reopens an empty completion drawer while typing', async ({}, testInfo) => {
  test.setTimeout(180_000)
  const fixture = await setupMockBackend()
  try {
    const { page } = fixture
    await waitForAppReady(fixture, 120_000)
    const editor = page.locator('[data-slot="composer-rich-input"]').first()
    await editor.fill('/goal @hermes-contributor in this screen capture')
    await editor.press('End')
    const drawer = page.locator('[data-slot="composer-completion-drawer"]')
    await expect(drawer).toHaveCount(0, { timeout: 20_000 })
    await page.evaluate(() => {
      const state = { openings: [] as { time: number; text: string }[], visible: false }
      const observer = new MutationObserver(() => {
        const menu = document.querySelector('[data-slot="composer-completion-drawer"]')
        if (menu && !state.visible) state.openings.push({ time: performance.now(), text: menu.textContent ?? '' })
        state.visible = !!menu
      })
      observer.observe(document.body, { subtree: true, childList: true, characterData: true })
      ;(window as any).__composerFlash = { state, observer }
    })
    // Real input events and real complete.slash RPCs; inference alone is mocked.
    await editor.pressSequentially(' we see the menu flash while typing', { delay: 180 })
    await expect(drawer).toHaveCount(0, { timeout: 20_000 })
    const openings = await page.evaluate(() => {
      const probe = (window as any).__composerFlash
      probe.observer.disconnect()
      return probe.state.openings
    })
    console.log('COMPOSER_FLASH_OPENINGS', JSON.stringify(openings))
    await testInfo.attach('menu-openings.json', {
      body: JSON.stringify(openings, null, 2),
      contentType: 'application/json'
    })
    await page.screenshot({ path: testInfo.outputPath('composer-prose.png') })
    expect(openings).toEqual([])
  } finally {
    await fixture.cleanup()
  }
})
