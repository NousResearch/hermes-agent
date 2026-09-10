import { expect, test } from './test'
import { setupMockBackend, waitForAppReady } from './fixtures'

// Uses the real panel, stylesheet, and native titlebar reservations. jsdom
// cannot detect clipped tabs or the content subtree's zero titlebar height.
test('sidebar tabs remain clickable while resizing on either window edge', async () => {
  const fixture = await setupMockBackend()
  try {
    await waitForAppReady(fixture, 120_000)
    const page = fixture.page
    const zone = page.locator('[data-tree-group]').filter({ has: page.locator('[data-tree-tab="sessions"]') })
    for (const flipped of [false, true]) {
      if (flipped) await page.getByRole('button', { name: 'Swap sidebar sides', exact: true }).click()
      for (const width of [245, 470]) {
        // Constrain the actual track, as a sash does, without coupling this
        // header regression to pointer-drag thresholds or persisted widths.
        await zone.evaluate((element, width) => {
          Object.assign(element.parentElement!.style, {
            flex: `0 0 ${width}px`,
            minWidth: `${width}px`,
            maxWidth: `${width}px`
          })
        }, width)
        for (const tab of [zone.locator('[data-tree-tab="sessions"]'), zone.getByRole('tab', { name: /^bots$/i })]) {
          await tab.click({ trial: true })
          const tabBox = (await tab.boundingBox())!
          const zoneBox = (await zone.boundingBox())!
          expect(tabBox.x).toBeGreaterThanOrEqual(zoneBox.x)
          expect(tabBox.x + tabBox.width).toBeLessThanOrEqual(zoneBox.x + zoneBox.width)
          if (width < 300) {
            const controlBox = (await page
              .locator(flipped ? '[aria-label="App controls"]' : '[aria-label="Window controls"]')
              .boundingBox())!
            expect(tabBox.y).toBeGreaterThanOrEqual(controlBox.y + controlBox.height)
          }
        }
        await zone.getByRole('button', { name: 'Minimize', exact: true }).click({ trial: true })
      }
    }
  } finally {
    await fixture.cleanup()
  }
})
