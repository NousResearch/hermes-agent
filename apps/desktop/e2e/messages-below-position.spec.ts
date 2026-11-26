import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'

import { build } from 'esbuild'

import { resolveElectronBinary } from './electron-binary'
import { _electron, expect, test } from './test'

test('messages-below reads stay bounded with real content-visibility groups', async ({}, testInfo) => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'hermes-messages-below-'))
  const main = path.join(temporary, 'main.cjs')
  writeFileSync(
    main,
    `const { app, BrowserWindow } = require('electron');
app.setPath('userData', ${JSON.stringify(path.join(temporary, 'user-data'))});
app.whenReady().then(() => {
  const window = new BrowserWindow({ width: 1000, height: 800,
    webPreferences: { sandbox: true, contextIsolation: true, backgroundThrottling: false } });
  window.loadURL('about:blank');
});`
  )

  const bundle = await build({
    entryPoints: [path.resolve('src/components/assistant-ui/thread/use-messages-below.ts')],
    bundle: true,
    format: 'iife',
    globalName: 'MessagesBelow',
    write: false
  })
  const app = await _electron.launch({
    executablePath: resolveElectronBinary([path.resolve('.'), path.resolve('../..')]),
    args: ['--no-sandbox', main]
  })

  try {
    const page = await app.firstWindow()
    await page.setContent('<div id="viewport" style="height:600px;overflow:auto"><div id="content"></div></div>')
    await page.addScriptTag({ content: bundle.outputFiles[0].text })
    const results = await page.evaluate(async () => {
      const count = 512
      const viewport = document.getElementById('viewport')!
      const content = document.getElementById('content')!
      const starts: number[] = []
      let top = 0

      for (let index = 0; index < count; index++) {
        starts.push(top)
        const height = 160 + (index % 3) * 40
        const group = document.createElement('div')
        group.dataset.slot = 'aui_message-group'
        group.style.cssText = `height:${height}px;content-visibility:auto;contain-intrinsic-size:auto ${height}px`
        group.innerHTML =
          '<div data-slot="aui_user-message-root">prompt</div><div data-slot="aui_assistant-message-root">answer</div>'
        content.append(group)
        top += height
      }

      const api = (window as unknown as {
        MessagesBelow: {
          createMessagesBelowReader: (
            viewport: HTMLElement,
            content: HTMLElement
          ) => { read: () => { count: number; settled: boolean } }
        }
      }).MessagesBelow
      const reader = api.createMessagesBelowReader(viewport, content)
      const original = Element.prototype.getBoundingClientRect
      let outer = 0
      let inner = 0
      Element.prototype.getBoundingClientRect = function () {
        if (this.getAttribute('data-slot') === 'aui_message-group') {
          outer++
        } else if (this.matches('[data-slot="aui_user-message-root"], [data-slot="aui_assistant-message-root"]')) {
          inner++
        }

        return original.call(this)
      }

      const measurements: Array<{
        target: number
        indexed: number
        reference: number
        outer: number
        inner: number
        referenceOuter: number
        referenceInner: number
      }> = []

      try {
        // Prime structural membership before measuring the scroll-time path.
        reader.read()

        for (const target of [0, 300, 30, 500, 200]) {
          viewport.scrollTop = starts[target] + 80
          await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
          outer = inner = 0
          const indexed = reader.read().count
          const measured = { target, indexed, reference: 0, outer, inner, referenceOuter: 0, referenceInner: 0 }
          measured.outer = outer
          measured.inner = inner
          outer = inner = 0
          const bottom = viewport.getBoundingClientRect().bottom
          let reference = 0

          for (const group of content.querySelectorAll<HTMLElement>('[data-slot="aui_message-group"]')) {
            const groupRect = group.getBoundingClientRect()

            if (groupRect.bottom <= bottom + 1) {
              continue
            }

            const messages = group.querySelectorAll<HTMLElement>(
              '[data-slot="aui_user-message-root"], [data-slot="aui_assistant-message-root"]'
            )

            if (groupRect.top >= bottom) {
              reference += messages.length
              continue
            }

            for (const message of messages) {
              const messageRect = message.getBoundingClientRect()

              if (messageRect.height > 0 && messageRect.bottom > bottom + 1) {
                reference++
              }
            }
          }

          measured.reference = reference
          measured.referenceOuter = outer
          measured.referenceInner = inner
          measurements.push(measured)
        }
      } finally {
        Element.prototype.getBoundingClientRect = original
      }

      return measurements
    })

    console.log('Messages-below geometry measurements:', JSON.stringify(results))

    for (const row of results) {
      expect(row.indexed).toBe(row.reference)
      expect(row.outer).toBeLessThanOrEqual(Math.ceil(Math.log2(512)) + 2)
      expect(row.inner).toBeLessThanOrEqual(2)
      expect(row.referenceOuter).toBe(512)
    }

    await testInfo.attach('messages-below-measurements.json', {
      body: JSON.stringify(results, null, 2),
      contentType: 'application/json'
    })
  } finally {
    await app.close()
    rmSync(temporary, { recursive: true, force: true })
  }
})
