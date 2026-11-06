import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'

import { build } from 'esbuild'

import { resolveElectronBinary } from './electron-binary'
import { _electron, expect, test } from './test'

const fixture = `
import { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { ModelEditSubmenu } from './src/app/shell/model-edit-submenu'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './src/components/ui/dialog'
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuSearch,
  DropdownMenuSub, DropdownMenuSubTrigger, DropdownMenuTrigger
} from './src/components/ui/dropdown-menu'

function Models() {
  const [effort, setEffort] = useState('medium')
  return <DropdownMenu defaultOpen>
    <DropdownMenuTrigger style={{ position: 'fixed', right: 40, bottom: 200 }}>Models</DropdownMenuTrigger>
    <DropdownMenuContent align="end" side="top" style={{ width: 280, padding: 0 }}>
      <DropdownMenuSearch aria-label="Search models" />
      <div id="models" style={{ height: 160, overflowY: 'auto' }}>
        {Array.from({ length: 30 }, (_, index) => <DropdownMenuSub key={index}>
          <DropdownMenuSubTrigger style={{ height: 32, boxSizing: 'border-box' }}>Model {index}</DropdownMenuSubTrigger>
          <ModelEditSubmenu defaultEffort="medium" effort={effort} fastControl={{ kind: 'none' }}
            isActive model={'model-' + index} provider="custom" reasoning
            onSelectModel={() => {}} onSetOptions={patch => setEffort(patch.effort ?? effort)} />
        </DropdownMenuSub>)}
      </div>
    </DropdownMenuContent>
  </DropdownMenu>
}
const dialog = document.getElementById('root').dataset.dialog === 'true'
createRoot(document.getElementById('root')).render(dialog ?
  <Dialog defaultOpen><DialogContent style={{ position: 'fixed', inset: 24 }}>
    <DialogTitle>Model settings</DialogTitle>
    <DialogDescription>Choose model options.</DialogDescription>
    <Models />
  </DialogContent></Dialog> : <Models />)
`

for (const dialog of [false, true]) {
  test(`model options hide when clipped and remain usable after returning (${dialog ? 'dialog' : 'body'})`, async () => {
    const temporary = mkdtempSync(path.join(tmpdir(), 'hermes-model-submenu-'))
    let app: Awaited<ReturnType<typeof _electron.launch>> | undefined

    try {
      const main = path.join(temporary, 'main.cjs')
      writeFileSync(
        main,
        `const { app, BrowserWindow } = require('electron');
app.setPath('userData', ${JSON.stringify(path.join(temporary, 'user-data'))});
app.whenReady().then(() => {
  const window = new BrowserWindow({ width: 1100, height: 900, show: false,
    webPreferences: { sandbox: true, contextIsolation: true, backgroundThrottling: false } });
  window.loadURL('about:blank');
});`
      )

      const bundle = await build({
        stdin: { contents: fixture, loader: 'tsx', resolveDir: path.resolve('.') },
        bundle: true,
        format: 'iife',
        jsx: 'automatic',
        define: { 'import.meta.env': '{}', 'import.meta.hot': 'undefined' },
        write: false
      })

      app = await _electron.launch({
        executablePath: resolveElectronBinary([path.resolve('.'), path.resolve('../..')]),
        args: ['--no-sandbox', main]
      })
      const page = await app.firstWindow()
      await page.setContent(`<div id="root" data-dialog="${dialog}"></div>`)
      await page.addScriptTag({ content: bundle.outputFiles[0].text })
      const list = page.locator('#models')
      const row = page.getByRole('menuitem', { name: 'Model 10', exact: true })
      const submenu = page.locator('[data-slot="dropdown-menu-sub-content"]')
      await list.evaluate(element => {
        element.scrollTop = 256
      })
      await row.hover()
      await expect(submenu).toBeVisible()
      expect(await submenu.evaluate(element => Boolean(element.closest('[data-slot="dialog-content"]')))).toBe(dialog)
      await page.mouse.move(10, 10)

      const alignment = () =>
        submenu.evaluate(element => {
          const trigger = document.querySelector('[data-slot="dropdown-menu-sub-trigger"][data-state="open"]')!

          return element.getBoundingClientRect().top - trigger.getBoundingClientRect().top
        })

      await expect.poll(alignment).toBeCloseTo(0, 0)
      await list.evaluate(element => {
        element.scrollTop += 24
      })
      await expect(submenu).toBeVisible()
      await expect.poll(alignment).toBeCloseTo(0, 0)

      // Keep the row inside the window: the list, rather than the viewport, clips it.
      for (const scrollTop of [512, 0]) {
        await list.evaluate((element, top) => {
          element.scrollTop = top
        }, scrollTop)
        const bounds = (await row.boundingBox())!
        expect(bounds.y).toBeGreaterThan(0)
        expect(bounds.y + bounds.height).toBeLessThan(await page.evaluate(() => innerHeight))
        await expect(submenu).toBeHidden()
        await list.evaluate(element => {
          element.scrollTop = 256
        })
        await expect(submenu).toBeVisible()
      }

      const high = page.getByRole('menuitemradio', { name: 'High', exact: true })
      await high.click()
      await expect(high).toBeChecked()
      await expect(submenu).toBeVisible()

      await submenu.press('ArrowLeft')
      await expect(submenu).toHaveCount(0)
      await row.press('ArrowRight')
      await expect(high).toBeVisible()
      await expect(high).toBeChecked()
    } finally {
      await app?.close()
      rmSync(temporary, { recursive: true, force: true })
    }
  })
}
