import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'

import type { CDPSession } from '@playwright/test'
import { createServer, type ViteDevServer } from 'vite'

import { resolveElectronBinary } from './electron-binary'
import { _electron, type ElectronApplication, expect, type Page, test } from './test'

// Exercise the production stylesheet with real layout and hit testing. The
// reduced DOM isolates the thread/dock geometry from sessions and inference.
let app: ElectronApplication | undefined
let server: ViteDevServer | undefined
let page: Page
let cdp: CDPSession
let css: string
let temporary: string

test.beforeAll(async () => {
  server = await createServer({ server: { middlewareMode: true }, appType: 'custom' })
  const stylesheet = await server.transformRequest('/src/styles.css?direct')
  expect(stylesheet).not.toBeNull()
  css = stylesheet!.code

  temporary = mkdtempSync(path.join(tmpdir(), 'hermes-touch-clearance-'))
  const main = path.join(temporary, 'main.cjs')
  writeFileSync(
    main,
    `const { app, BrowserWindow } = require('electron');
app.setPath('userData', ${JSON.stringify(path.join(temporary, 'user-data'))});
app.whenReady().then(() => {
  const window = new BrowserWindow({ width: 1000, height: 800,
    webPreferences: { sandbox: true, contextIsolation: true } });
  window.loadURL('about:blank');
});`
  )
  app = await _electron.launch({
    executablePath: resolveElectronBinary([path.resolve('.'), path.resolve('../..')]),
    args: ['--no-sandbox', main]
  })
  page = await app.firstWindow()
  cdp = await page.context().newCDPSession(page)
})

test.afterAll(async () => {
  await app?.close()
  await server?.close()

  if (temporary) {
    rmSync(temporary, { recursive: true, force: true })
  }
})

function surface(id: string, dockHeight: number): string {
  // The 8px measurement buckets can round a dock down by almost 4px.
  const measured = Math.round(dockHeight / 8) * 8

  return `<section data-chat-surface data-testid="${id}" style="position:relative;height:100%;width:50%;
    --composer-measured-height:${measured}px;--composer-surface-measured-height:80px">
    <div style="height:100%;overflow:hidden">
      <div style="height:var(--thread-viewport-height);overflow:hidden">
        <div data-slot="aui_thread-viewport" class="size-full overflow-x-hidden overflow-y-auto overscroll-contain">
          <div data-slot="aui_thread-content" class="mx-auto flex min-h-full w-full min-w-0 flex-col px-6">
            <div style="height:1500px;flex:none;padding-top:1000px">
              <button type="button" style="height:44px;width:80px;background:royalblue;color:white"
                onclick="this.dataset.clicked='true'">Action</button>
            </div>
            <div data-slot="aui_composer-clearance" class="shrink-0"
              style="height:var(--thread-last-message-clearance)"></div>
          </div>
        </div>
      </div>
    </div>
    <div data-slot="composer-dock" class="absolute bottom-0 left-1/2 -translate-x-1/2 z-30 flex flex-col"
      style="height:${dockHeight}px;background:slategray">
      <div style="flex:1">Status</div><div style="height:80px;background:navy;color:white">Input</div>
    </div>
  </section>`
}

async function showSurfaces({
  browser,
  touch,
  heights
}: {
  browser: boolean
  touch: boolean
  heights: [number, number]
}) {
  await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: touch, maxTouchPoints: 1 })
  await page.setContent(`<html ${browser ? 'data-hermes-desktop-host="browser"' : ''}>
    <body><main style="display:flex;width:100vw;height:100vh">
      ${surface('left', heights[0])}${surface('right', heights[1])}
    </main></body></html>`)
  await page.addStyleTag({ content: css })
  expect(await page.evaluate(() => matchMedia('(pointer: coarse)').matches)).toBe(touch)
}

test('touch transcript controls clear each dock after scrolling, resizing and composer growth', async () => {
  const testInfo = test.info()

  for (const size of [
    { width: 393, height: 700 },
    { width: 393, height: 380 },
    { width: 844, height: 393 }
  ]) {
    await app!.evaluate(({ BrowserWindow }, dimensions) => {
      BrowserWindow.getAllWindows()[0].setContentSize(dimensions.width, dimensions.height)
    }, size)
    await expect.poll(() => page.evaluate(() => innerHeight)).toBe(size.height)

    for (const heights of [
      [123, 243],
      [203, 283]
    ] as [number, number][]) {
      await showSurfaces({ browser: true, touch: true, heights })

      for (const id of ['left', 'right']) {
        const pane = page.getByTestId(id)
        const action = pane.getByRole('button', { name: 'Action' })
        await action.evaluate(element => element.scrollIntoView({ block: 'end' }))

        const geometry = await pane.evaluate(element => {
          const viewport = element.querySelector('[data-slot="aui_thread-viewport"]')!.getBoundingClientRect()
          const dock = element.querySelector('[data-slot="composer-dock"]')!.getBoundingClientRect()
          const button = element.querySelector('button')!
          const rect = button.getBoundingClientRect()

          return {
            viewportBottom: viewport.bottom,
            dockTop: dock.top,
            clickable: button.contains(document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2))
          }
        })

        expect(geometry.viewportBottom).toBeLessThanOrEqual(geometry.dockTop)
        expect(geometry.clickable).toBe(true)
        await action.click({ timeout: 2000 })
        await expect(action).toHaveAttribute('data-clicked', 'true')
      }
    }
  }

  // The floating composer is portaled outside the pane and publishes zero.
  await showSurfaces({ browser: true, touch: true, heights: [0, 0] })
  await page
    .locator('[data-slot="composer-dock"]')
    .evaluateAll(elements => elements.forEach(element => element.remove()))

  for (const id of ['left', 'right']) {
    const sizes = await page.getByTestId(id).evaluate(element => ({
      pane: element.clientHeight,
      viewport: element.querySelector('[data-slot="aui_thread-viewport"]')!.clientHeight
    }))

    expect(sizes.viewport).toBe(sizes.pane)
  }

  await page.screenshot({ path: testInfo.outputPath('touch-composer-clearance.png') })
})

test('native Desktop and mouse-driven browser keep their overlay layout', async () => {
  for (const mode of [
    { browser: false, touch: true },
    { browser: true, touch: false }
  ]) {
    await showSurfaces({ ...mode, heights: [123, 243] })

    const geometry = await page.getByTestId('right').evaluate(element => ({
      viewportBottom: element.querySelector('[data-slot="aui_thread-viewport"]')!.getBoundingClientRect().bottom,
      dockTop: element.querySelector('[data-slot="composer-dock"]')!.getBoundingClientRect().top
    }))

    expect(geometry.viewportBottom).toBeGreaterThan(geometry.dockTop)
  }
})
