import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'

import { build } from 'esbuild'
import { createServer, type ViteDevServer } from 'vite'

import { resolveElectronBinary } from './electron-binary'
import { _electron, type ElectronApplication, expect, type Page, test } from './test'

let app: ElectronApplication | undefined
let server: ViteDevServer | undefined
let page: Page
let css: string
let script: string
let temporary: string

test.beforeAll(async () => {
  server = await createServer({
    server: { middlewareMode: true },
    appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] }
  })
  const stylesheet = await server.transformRequest('/src/styles.css?direct')
  expect(stylesheet).not.toBeNull()
  css = stylesheet!.code

  // Use the real approval-jump action and session stores. Only the transcript
  // and dock DOM are reduced to isolate browser geometry from inference.
  const bundle = await build({
    stdin: {
      contents: `
import React from 'react';
import { createRoot } from 'react-dom/client';
import { ScrollToBottomButton } from './app/chat/scroll-to-bottom-button';
import { ComposerSurfaceProvider } from './app/chat/composer/scope';
import { setApprovalRequest } from './store/prompts';
import { setThreadAtBottom } from './store/thread-scroll';
export function mount() {
  for (const id of ['left', 'right']) {
    setApprovalRequest({ sessionId: id, command: 'echo test', description: 'Test approval' });
    setThreadAtBottom(false, id);
    createRoot(document.querySelector('[data-jump="' + id + '"]')).render(
      React.createElement(ComposerSurfaceProvider, { value: id },
        React.createElement(ScrollToBottomButton, { sessionId: id })));
  }
}`,
      resolveDir: path.resolve('src'),
      loader: 'tsx'
    },
    tsconfig: path.resolve('tsconfig.json'),
    bundle: true,
    format: 'iife',
    globalName: 'NativeComposerProbe',
    define: { 'import.meta.env': '{}', 'import.meta.hot': 'undefined', 'process.env.NODE_ENV': '"development"' },
    write: false
  })

  script = bundle.outputFiles[0].text

  temporary = mkdtempSync(path.join(tmpdir(), 'hermes-native-reveal-'))
  const main = path.join(temporary, 'main.cjs')
  writeFileSync(
    main,
    `const { app, BrowserWindow } = require('electron');
app.setPath('userData', ${JSON.stringify(path.join(temporary, 'user-data'))});
app.whenReady().then(() => {
  const window = new BrowserWindow({ width: 1100, height: 700,
    webPreferences: { sandbox: true, contextIsolation: true } });
  window.loadURL('about:blank');
});`
  )
  app = await _electron.launch({
    executablePath: resolveElectronBinary([path.resolve('.'), path.resolve('../..')]),
    args: ['--no-sandbox', main]
  })
  page = await app.firstWindow()
  await page.emulateMedia({ reducedMotion: 'reduce' })
})

test.afterAll(async () => {
  await app?.close()
  await server?.close()

  if (temporary) {
    rmSync(temporary, { recursive: true, force: true })
  }
})

function surface(id: string, height: number): string {
  return `<section data-chat-surface data-testid="${id}" style="position:relative;width:50%;height:100%;
    --composer-measured-height:${Math.round(height / 8) * 8}px;--composer-surface-measured-height:80px">
    <div style="height:var(--thread-viewport-height);overflow:hidden">
      <div data-slot="aui_thread-viewport" class="size-full overflow-y-auto">
        <div style="height:2400px;position:relative">
          <div data-approval-stack data-session-id="${id}" style="position:absolute;top:1200px;height:72px">
            <p>Approval request</p>
            <button type="button" style="height:44px;width:80px;background:royalblue;color:white"
              onclick="this.dataset.clicked='true'">Run</button>
          </div>
          <div data-tail style="position:absolute;top:2360px;height:24px">Final reply</div>
        </div>
        <div data-slot="aui_composer-clearance" style="height:var(--thread-last-message-clearance)"></div>
      </div>
    </div>
    <div data-slot="composer-dock" class="absolute bottom-0 left-1/2 -translate-x-1/2 z-30 flex flex-col"
      style="height:${height}px;background:slategray"><div style="flex:1">Status</div><div>Input</div></div>
    <div data-jump="${id}"></div>
  </section>`
}

async function showSurfaces(heights: [number, number]) {
  await page.setContent(`<html><body><main style="display:flex;height:100vh;width:100vw">
    ${surface('left', heights[0])}${surface('right', heights[1])}
  </main></body></html>`)
  await page.addStyleTag({ content: css })
  await page.addScriptTag({ content: script })
  await page.evaluate(() =>
    (window as unknown as { NativeComposerProbe: { mount: () => void } }).NativeComposerProbe.mount()
  )
  await expect(page.getByRole('button', { name: 'Approval needed' })).toHaveCount(2)
}

async function parkApproval(id: string) {
  return page.getByTestId(id).evaluate(element => {
    const viewport = element.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!
    const approval = element.querySelector<HTMLElement>('[data-approval-stack]')!
    // Wholly inside the browser's viewport, but behind its overlay composer.
    viewport.scrollTop = approval.offsetTop + approval.offsetHeight - viewport.clientHeight + 8

    return viewport.scrollTop
  })
}

test('the native approval jump reveals tappable controls above each composer', async () => {
  for (const size of [
    { width: 1100, height: 700 },
    { width: 800, height: 500 }
  ]) {
    await app!.evaluate(({ BrowserWindow }, dimensions) => {
      BrowserWindow.getAllWindows()[0].setContentSize(dimensions.width, dimensions.height)
    }, size)
    await expect.poll(() => page.evaluate(() => innerHeight)).toBe(size.height)

    for (const heights of [
      [123, 243],
      [203, 283]
    ] as [number, number][]) {
      await showSurfaces(heights)

      for (const id of ['left', 'right']) {
        await parkApproval(id)
        const pane = page.getByTestId(id)
        const before = await pane.locator('[data-slot="aui_thread-viewport"]').evaluate(element => element.scrollHeight)
        await pane.getByRole('button', { name: 'Approval needed' }).click()

        const geometry = await pane.getByRole('button', { name: 'Run', exact: true }).evaluate(element => {
          const rect = element.getBoundingClientRect()
          const dock = element.closest('[data-chat-surface]')!.querySelector('[data-slot="composer-dock"]')!

          return {
            buttonBottom: rect.bottom,
            dockTop: dock.getBoundingClientRect().top,
            clickable: element.contains(document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2))
          }
        })

        expect(geometry.buttonBottom).toBeLessThanOrEqual(geometry.dockTop)
        expect(geometry.clickable).toBe(true)
        await pane.getByRole('button', { name: 'Run', exact: true }).click({ timeout: 2000 })
        await expect(pane.getByRole('button', { name: 'Run', exact: true })).toHaveAttribute('data-clicked', 'true')
        expect(await pane.locator('[data-slot="aui_thread-viewport"]').evaluate(element => element.scrollHeight)).toBe(
          before
        )

        // Ordinary bottom-follow still uses the existing trailing clearance.
        const tail = await pane.evaluate(element => {
          const viewport = element.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!
          viewport.scrollTop = viewport.scrollHeight

          return {
            bottom: element.querySelector('[data-tail]')!.getBoundingClientRect().bottom,
            dockTop: element.querySelector('[data-slot="composer-dock"]')!.getBoundingClientRect().top
          }
        })

        expect(tail.bottom).toBeLessThanOrEqual(tail.dockTop)
      }
    }
  }
})

test('HUD, detached, absent and browser composers do not reserve a native reveal inset', async () => {
  for (const mode of ['hud', 'detached', 'absent', 'browser']) {
    await showSurfaces([123, 243])
    await page.evaluate(current => {
      if (current === 'browser') {
        document.documentElement.dataset.hermesDesktopHost = 'browser'
      }

      if (current === 'hud') {
        document.querySelector('main')!.setAttribute('data-hud-shell', '')
      }

      for (const dock of document.querySelectorAll('[data-slot="composer-dock"]')) {
        if (current === 'detached') {
          dock.setAttribute('data-popped-out', '')
        }

        if (current === 'absent') {
          dock.remove()
        }
      }
    }, mode)
    const before = await parkApproval('left')
    await page
      .getByTestId('left')
      .locator('[data-approval-stack]')
      .evaluate(element => element.scrollIntoView({ block: 'nearest' }))

    const after = await page
      .getByTestId('left')
      .locator('[data-slot="aui_thread-viewport"]')
      .evaluate(element => element.scrollTop)

    expect(after).toBe(before)
  }
})
