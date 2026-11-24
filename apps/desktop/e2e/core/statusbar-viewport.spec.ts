import { fileURLToPath } from 'node:url'

import { type Browser, chromium, expect, type Locator, type Page, test } from '@playwright/test'
import { createServer, type ViteDevServer } from 'vite'

const desktopRoot = fileURLToPath(new URL('../..', import.meta.url))
let server: ViteDevServer
let browser: Browser
let url: string

test.beforeAll(async () => {
  server = await createServer({
    cacheDir: `${desktopRoot}/node_modules/.cache/statusbar-viewport`,
    configFile: `${desktopRoot}/vite.config.ts`,
    root: desktopRoot,
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    plugins: [
      {
        name: 'statusbar-viewport-fixture',
        configureServer(vite) {
          vite.middlewares.use('/statusbar-viewport-test', async (_request, response, next) => {
            try {
              response.setHeader('Content-Type', 'text/html')
              response.end(
                await vite.transformIndexHtml(
                  '/statusbar-viewport-test',
                  `<!doctype html>
<html lang="en" data-hermes-desktop-host="browser" style="height:100%">
<head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="margin:0;height:100%"><div id="root" style="height:100%"></div>
<script type="module" src="/src/test/statusbar-viewport.fixture.tsx"></script></body></html>`
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
  url = `${server.resolvedUrls!.local[0]}statusbar-viewport-test`
  browser = await chromium.launch({ executablePath: process.env.HERMES_E2E_CHROMIUM || undefined })
})

test.afterAll(async () => {
  await browser?.close()
  await server?.close()
})

async function setScale(page: Page, percent: number) {
  await page.evaluate(value => {
    ;(window as unknown as { setTestZoom: (percent: number) => void }).setTestZoom(value)
  }, percent)
}

async function expectContained(menu: Locator, page: Page) {
  await expect
    .poll(
      async () => {
        const rect = await menu.boundingBox()
        const viewport = page.viewportSize()!

        return Boolean(
          rect &&
          rect.x >= -1 &&
          rect.y >= -1 &&
          rect.x + rect.width <= viewport.width + 1 &&
          rect.y + rect.height <= viewport.height + 1
        )
      },
      { message: 'menu fits inside the visible viewport', timeout: 5000 }
    )
    .toBe(true)
}

test('context usage stays anchored and Compact conversation remains reachable at browser UI scales', async () => {
  const page = await browser.newPage({ viewport: { width: 1366, height: 768 }, reducedMotion: 'reduce' })

  try {
    await page.goto(url)
    const trigger = page.getByRole('button', { name: 'Context meter', exact: true })
    await expect(trigger).toBeVisible()

    for (const percent of [100, 125, 150, 80, 200]) {
      await setScale(page, percent)
      await trigger.click()
      await expectContained(page.locator('[data-slot="dropdown-menu-content"]'), page)
      await page.keyboard.press('Escape')
    }

    // A browser/OS zoom reduces the CSS viewport independently of app UI Scale.
    await setScale(page, 100)
    await page.setViewportSize({ width: 1093, height: 614 })
    await trigger.click()
    const menu = page.locator('[data-slot="dropdown-menu-content"]')
    await expectContained(menu, page)

    // Resize and change scale with the menu open, then reach the bottom action.
    await page.setViewportSize({ width: 800, height: 320 })
    await setScale(page, 125)
    await expectContained(menu, page)
    expect(await menu.evaluate(element => element.scrollHeight > element.clientHeight)).toBe(true)
    await page.getByRole('button', { name: 'Compact conversation', exact: true }).click()
    await expect(page.locator('body')).toHaveAttribute('data-compacted', 'true')
    await expect(menu).toHaveCount(0)
  } finally {
    await page.close()
  }
})

test('statusbar visibility menu scrolls to its final command in a scaled short viewport', async () => {
  const page = await browser.newPage({ viewport: { width: 1366, height: 768 }, reducedMotion: 'reduce' })

  try {
    await page.goto(url)
    const statusbar = page.getByRole('contentinfo')
    await expect(statusbar).toBeVisible()

    for (const percent of [100, 125, 150, 80, 200]) {
      await setScale(page, percent)
      await statusbar.click({ button: 'right', position: { x: 10, y: 5 } })
      await expectContained(page.locator('[data-slot="context-menu-content"]'), page)
      await page.keyboard.press('Escape')
    }

    await page.setViewportSize({ width: 800, height: 320 })
    await setScale(page, 125)
    await statusbar.click({ button: 'right', position: { x: 10, y: 5 } })
    const menu = page.locator('[data-slot="context-menu-content"]')
    await expectContained(menu, page)
    expect(await menu.evaluate(element => element.scrollHeight > element.clientHeight)).toBe(true)
    await page.keyboard.press('End')
    await expect(page.getByRole('menuitem', { name: /Hide status bar/ })).toBeFocused()
    await expectContained(page.getByRole('menuitem', { name: /Hide status bar/ }), page)
    await page.keyboard.press('Escape')
    await expect(menu).toHaveCount(0)
  } finally {
    await page.close()
  }
})
