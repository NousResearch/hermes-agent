import { fileURLToPath } from 'node:url'

import { type Browser, chromium, expect, type Locator, type Page, test } from '@playwright/test'
import { createServer, type ViteDevServer } from 'vite'

const desktopRoot = fileURLToPath(new URL('../..', import.meta.url))
let server: ViteDevServer
let browser: Browser
let url: string

test.beforeAll(async () => {
  server = await createServer({
    cacheDir: `${desktopRoot}/node_modules/.cache/webapp-scaled-poppers`,
    configFile: `${desktopRoot}/vite.config.ts`,
    root: desktopRoot,
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    plugins: [
      {
        name: 'webapp-scaled-poppers-fixture',
        configureServer(vite) {
          vite.middlewares.use('/webapp-scaled-poppers-test', async (_request, response, next) => {
            try {
              response.setHeader('Content-Type', 'text/html')
              response.end(
                await vite.transformIndexHtml(
                  '/webapp-scaled-poppers-test',
                  `<!doctype html>
<html lang="en" data-hermes-desktop-host="browser" style="height:100%">
<head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="margin:0;height:100%;overflow:hidden"><div id="root" style="height:100%"></div>
<script type="module" src="/src/test/webapp-scaled-poppers.fixture.tsx"></script></body></html>`
                )
              )
            } catch (error) {
              next(error)
            }
          })
        }
      }
    ]
  })
  await server.listen()
  url = `${server.resolvedUrls!.local[0]}webapp-scaled-poppers-test`
  browser = await chromium.launch({ executablePath: process.env.HERMES_E2E_CHROMIUM || undefined })
})

test.afterAll(async () => {
  await browser?.close()
  await server?.close()
})

async function setScale(page: Page, percent: number) {
  await page.evaluate(async value => {
    ;(window as unknown as { setTestZoom: (percent: number) => void }).setTestZoom(value)
    await new Promise<void>(resolve => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    })
  }, percent)
}

async function expectContained(locator: Locator, page: Page, label = 'surface') {
  await expect
    .poll(
      async () => {
        const rect = await locator.boundingBox()
        const viewport = page.viewportSize()!

        return Boolean(
          rect &&
          rect.x >= -1 &&
          rect.y >= -1 &&
          rect.x + rect.width <= viewport.width + 1 &&
          rect.y + rect.height <= viewport.height + 1
        )
      },
      { message: `${label} fits inside the visible viewport`, timeout: 5000 }
    )
    .toBe(true)
}

test('profile dropdown stays inside the viewport and scrolls via mouse wheel and touch at scaled Webapp UI', async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 700 }, reducedMotion: 'reduce' })

  try {
    await page.goto(url)
    const trigger = page.locator('[data-slot="profile-dropdown"]').first()
    await expect(trigger).toBeVisible()

    for (const percent of [80, 100, 110, 125, 150, 200]) {
      await setScale(page, percent)
      await trigger.click()
      const menu = page.locator('[data-slot="dropdown-menu-content"]')
      await expectContained(menu, page, `profile dropdown at ${percent}%`)
      await page.keyboard.press('Escape')
      await expect(menu).toHaveCount(0)
    }

    // 125% scale: 22 profiles exceed the available height above the sidebar foot,
    // so the menu must clamp and scroll via mouse wheel to reach the final profile.
    await setScale(page, 125)
    await trigger.click()
    const menu = page.locator('[data-slot="dropdown-menu-content"]')
    await expectContained(menu, page, 'profile dropdown at 125%')
    expect(await menu.evaluate(el => el.scrollHeight > el.clientHeight)).toBe(true)

    const menuBox = (await menu.boundingBox())!
    await page.mouse.move(menuBox.x + menuBox.width / 2, menuBox.y + menuBox.height / 2)
    await page.mouse.wheel(0, 600)
    await expect.poll(() => menu.evaluate(el => el.scrollTop), { timeout: 5000 }).toBeGreaterThan(0)
    await expectContained(page.getByRole('menuitemradio', { name: 'zulu' }), page, 'scrolled zulu row')
  } finally {
    await page.close()
  }

  // Android tablet / touch viewport: touch swipe must also scroll the clamped profile dropdown.
  const touchContext = await browser.newContext({
    hasTouch: true,
    isMobile: true,
    reducedMotion: 'reduce',
    viewport: { width: 1024, height: 650 }
  })

  const touchPage = await touchContext.newPage()

  try {
    await touchPage.goto(url)
    await setScale(touchPage, 125)
    await touchPage.locator('[data-slot="profile-dropdown"]').first().click()
    const menu = touchPage.locator('[data-slot="dropdown-menu-content"]')
    await expectContained(menu, touchPage, 'touch profile dropdown at 125%')
    expect(await menu.evaluate(el => el.scrollHeight > el.clientHeight)).toBe(true)

    const box = (await menu.boundingBox())!
    const cx = box.x + box.width / 2
    const startY = box.y + box.height - 40
    const cdp = await touchContext.newCDPSession(touchPage)

    await cdp.send('Input.dispatchTouchEvent', { touchPoints: [{ id: 1, x: cx, y: startY }], type: 'touchStart' })

    for (let step = 1; step <= 10; step++) {
      await cdp.send('Input.dispatchTouchEvent', {
        touchPoints: [{ id: 1, x: cx, y: startY - step * 24 }],
        type: 'touchMove'
      })
    }

    await cdp.send('Input.dispatchTouchEvent', { touchPoints: [], type: 'touchEnd' })
    await expect.poll(() => menu.evaluate(el => el.scrollTop), { timeout: 5000 }).toBeGreaterThan(0)
    await expectContained(touchPage.getByRole('menuitemradio', { name: 'zulu' }), touchPage, 'touch-scrolled zulu')
  } finally {
    await touchContext.close()
  }
})

test('model picker aligns to its right-edge trigger, scrolls, and keeps submenus inside the viewport', async () => {
  const page = await browser.newPage({ viewport: { width: 1200, height: 650 }, reducedMotion: 'reduce' })

  try {
    await page.goto(url)
    const trigger = page.getByTestId('model-pill')
    await expect(trigger).toBeVisible()

    for (const percent of [80, 100, 110, 125, 150, 200]) {
      await setScale(page, percent)
      const triggerBox = (await trigger.boundingBox())!
      await trigger.click()
      const menu = page.locator('[data-slot="dropdown-menu-content"]')
      await expectContained(menu, page, `model picker at ${percent}%`)

      const menuBox = (await menu.boundingBox())!
      // align="end" anchors the menu's right edge to the trigger's right edge (never shifted past it).
      expect(Math.abs(menuBox.x + menuBox.width - (triggerBox.x + triggerBox.width))).toBeLessThanOrEqual(4)
      // side="top" keeps the menu above its trigger rather than dropping over the composer/statusbar.
      expect(menuBox.y + menuBox.height).toBeLessThanOrEqual(triggerBox.y + 1)

      await page.keyboard.press('Escape')
      await expect(menu).toHaveCount(0)
    }

    // At 125% scale, scroll to the submenu trigger, open it, and pick an item inside the viewport.
    await setScale(page, 125)
    await trigger.click()
    const menu = page.locator('[data-slot="dropdown-menu-content"]')
    await expectContained(menu, page, 'model picker at 125%')
    expect(await menu.evaluate(el => el.scrollHeight > el.clientHeight)).toBe(true)

    const subTrigger = page.getByTestId('model-sub-trigger')
    await subTrigger.scrollIntoViewIfNeeded()
    await subTrigger.hover()
    const submenu = page.locator('[data-slot="dropdown-menu-sub-content"]')
    await expect(submenu).toBeVisible()
    await expectContained(submenu, page, 'model submenu at 125%')
    // Enter the submenu horizontally from the sub-trigger (matching real pointer travel)
    // before clicking an upper row, so the pointer doesn't sweep over sibling parent items.
    const subBox = (await submenu.boundingBox())!
    const subTriggerBox = (await subTrigger.boundingBox())!
    await page.mouse.move(subBox.x + subBox.width / 2, subTriggerBox.y + subTriggerBox.height / 2, { steps: 6 })
    await page.getByRole('menuitem', { name: 'Submodel 1', exact: true }).click()
    await expect(page.locator('body')).toHaveAttribute('data-submodel-picked', 'Submodel 1')
  } finally {
    await page.close()
  }
})

test('Select width, Popover and Tooltip arrows, and click-anchored AppContextMenu stay aligned at every scale', async () => {
  const page = await browser.newPage({ viewport: { width: 1200, height: 650 }, reducedMotion: 'reduce' })

  try {
    await page.goto(url)

    for (const percent of [80, 100, 125, 150, 200]) {
      await setScale(page, percent)

      // Select: width matches trigger width (not scaled a second time) and stays inside viewport.
      const selectTrigger = page.getByTestId('body-select')
      const selectTriggerBox = (await selectTrigger.boundingBox())!
      await selectTrigger.click()
      const selectContent = page.locator('[data-slot="select-content"]')
      await expectContained(selectContent, page, `Select at ${percent}%`)
      const selectBox = (await selectContent.boundingBox())!
      expect(Math.abs(selectBox.width - selectTriggerBox.width)).toBeLessThanOrEqual(6)
      await page.keyboard.press('Escape')
      await expect(selectContent).toHaveCount(0)

      // Popover with arrow (bottom-right, align="end", side="top"): arrow stays visible and points at trigger center.
      const brPopoverTrigger = page.getByTestId('br-popover')
      const brPopoverTriggerBox = (await brPopoverTrigger.boundingBox())!
      await brPopoverTrigger.click()
      const popover = page.locator('[data-slot="popover-content"]')
      await expectContained(popover, page, `BR Popover at ${percent}%`)

      const popoverArrow = await popover.evaluate(el => {
        const span = [...el.children].find(
          child => child.tagName === 'SPAN' && getComputedStyle(child).position === 'absolute'
        ) as HTMLElement | undefined

        const rect = span?.getBoundingClientRect()

        return {
          cx: rect ? rect.x + rect.width / 2 : null,
          hidden: span?.style.visibility === 'hidden'
        }
      })

      expect(popoverArrow.hidden).toBe(false)
      expect(Math.abs(popoverArrow.cx! - (brPopoverTriggerBox.x + brPopoverTriggerBox.width / 2))).toBeLessThanOrEqual(
        3
      )
      await page.keyboard.press('Escape')
      await expect(popover).toHaveCount(0)

      // Tooltip with arrow (bottom-right): stays inside viewport and arrow aligns with trigger center.
      const brTipTrigger = page.getByTestId('br-tip')
      const brTipTriggerBox = (await brTipTrigger.boundingBox())!
      await brTipTrigger.hover()
      const tooltip = page.locator('[data-slot="tooltip-content"]')
      await expectContained(tooltip, page, `BR Tooltip at ${percent}%`)

      const tipArrow = await tooltip.evaluate(el => {
        const span = [...el.children].find(
          child => child.tagName === 'SPAN' && getComputedStyle(child).position === 'absolute'
        ) as HTMLElement | undefined

        const rect = span?.getBoundingClientRect()

        return {
          cx: rect ? rect.x + rect.width / 2 : null,
          hidden: span?.style.visibility === 'hidden'
        }
      })

      expect(tipArrow.hidden).toBe(false)
      expect(Math.abs(tipArrow.cx! - (brTipTriggerBox.x + brTipTriggerBox.width / 2))).toBeLessThanOrEqual(3)
      await page.mouse.move(0, 0)
    }

    // Non-modal Popover content stays clickable inside the pointer-events: none browser portal host.
    await setScale(page, 125)
    await page.getByTestId('body-popover').click()
    await page.getByTestId('popover-action').click()
    await expect(page.locator('body')).toHaveAttribute('data-popover-clicked', 'true')
    await page.keyboard.press('Escape')

    // Click-anchored AppContextMenu opens at the actual right-click point (not multiplied by zoom).
    const clickX = 880
    const clickY = 260
    await page.mouse.click(clickX, clickY, { button: 'right' })
    const ctxMenu = page.locator('[data-slot="dropdown-menu-content"]')
    await expectContained(ctxMenu, page, 'AppContextMenu at 125%')
    const ctxBox = (await ctxMenu.boundingBox())!
    expect(Math.abs(ctxBox.x - clickX)).toBeLessThanOrEqual(12)
    expect(Math.abs(ctxBox.y - clickY)).toBeLessThanOrEqual(16)
    await page.keyboard.press('Escape')
  } finally {
    await page.close()
  }
})

test('Dialog-nested, BootFailureModal-nested, and mobile Sheet poppers stay aligned and interactive at 125% scale', async () => {
  const page = await browser.newPage({ viewport: { width: 1200, height: 650 }, reducedMotion: 'reduce' })

  try {
    await page.goto(url)
    await setScale(page, 125)

    // Dialog-nested Select & Popover: no double-division inside transformed DialogContent.
    await page.getByTestId('open-dialog').click()
    const dialogSelectTrigger = page.getByTestId('dialog-select')
    const dialogSelectTriggerBox = (await dialogSelectTrigger.boundingBox())!
    await dialogSelectTrigger.click()
    const dialogSelect = page.locator('[data-slot="select-content"]')
    await expectContained(dialogSelect, page, 'Dialog Select at 125%')
    const dialogSelectBox = (await dialogSelect.boundingBox())!
    expect(Math.abs(dialogSelectBox.x - dialogSelectTriggerBox.x)).toBeLessThanOrEqual(4)
    expect(Math.abs(dialogSelectBox.width - dialogSelectTriggerBox.width)).toBeLessThanOrEqual(6)
    await page.keyboard.press('Escape')
    await page.keyboard.press('Escape')

    // BootFailureModal-nested Select (fixed inset-0 [contain:layout]): aligned and contained.
    await page.getByTestId('open-boot-modal').click()
    const bootSelectTrigger = page.getByTestId('boot-select')
    const bootSelectTriggerBox = (await bootSelectTrigger.boundingBox())!
    await bootSelectTrigger.click()
    const bootSelect = page.locator('[data-slot="select-content"]')
    await expectContained(bootSelect, page, 'BootFailureModal Select at 125%')
    const bootSelectBox = (await bootSelect.boundingBox())!
    expect(Math.abs(bootSelectBox.x - bootSelectTriggerBox.x)).toBeLessThanOrEqual(4)
    expect(Math.abs(bootSelectBox.width - bootSelectTriggerBox.width)).toBeLessThanOrEqual(6)
    await page.keyboard.press('Escape')
  } finally {
    await page.close()
  }

  // Mobile Sheet drawer (z-50): ProfileDropdown inside browser portal host (z-index: 60) paints above the drawer.
  const sheetPage = await browser.newPage({ viewport: { width: 760, height: 650 }, reducedMotion: 'reduce' })

  try {
    await sheetPage.goto(url)
    await setScale(sheetPage, 125)
    await sheetPage.getByTestId('open-sheet').click()
    const sheetTrigger = sheetPage.getByTestId('sheet-profile-rail').locator('[data-slot="profile-dropdown"]')
    await expect(sheetTrigger).toBeVisible()
    await sheetTrigger.click()
    const menu = sheetPage.locator('[data-slot="dropdown-menu-content"]')
    await expectContained(menu, sheetPage, 'Sheet ProfileDropdown at 125%')

    // Hit-test the center of the menu to prove it paints above the z-50 Sheet overlay/content.
    const hitInsideMenu = await menu.evaluate(el => {
      const rect = el.getBoundingClientRect()
      const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2)

      return Boolean(hit && el.contains(hit))
    })

    expect(hitInsideMenu).toBe(true)
  } finally {
    await sheetPage.close()
  }
})
