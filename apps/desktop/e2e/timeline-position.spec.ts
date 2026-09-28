import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'

import { build } from 'esbuild'

import { _electron, expect, test } from './test'

/** Native Chromium geometry, without a backend or the app's other scroll observers. */
test('timeline reads stay logarithmic with real skipped tool-heavy groups', async ({}, testInfo) => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'hermes-timeline-position-'))
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
    entryPoints: [path.resolve('src/components/assistant-ui/thread/timeline-position.ts')],
    bundle: true,
    format: 'iife',
    globalName: 'TimelinePosition',
    write: false
  })
  const app = await _electron.launch({ args: ['--no-sandbox', main] })

  try {
    const page = await app.firstWindow()
    await page.setContent('<div id="viewport" style="height:600px;overflow:auto"></div>')
    await page.addScriptTag({ content: bundle.outputFiles[0].text })
    const results = await page.evaluate(async () => {
      const count = 512
      const viewport = document.getElementById('viewport')!
      const starts: number[] = []
      let top = 0

      for (let index = 0; index < count; index++) {
        starts.push(top)
        const height = 400 + (index % 5) * 37
        const group = document.createElement('div')
        group.dataset.slot = 'aui_message-group'
        group.style.cssText = `height:${height}px;content-visibility:auto;contain-intrinsic-size:auto ${height}px`
        group.innerHTML = `<div data-slot="aui_turn-pair"><div data-message-id="u${index}">Prompt ${index}</div>${'<p>Tool output and rendered markdown paragraph.</p>'.repeat(20)}</div>`
        viewport.append(group)
        top += height
      }

      const api = (window as unknown as {
        TimelinePosition: {
          createTimelinePositionReader: (viewport: HTMLElement, indexes: Map<string, number>) => { read: () => number }
        }
      }).TimelinePosition
      const reader = api.createTimelinePositionReader(viewport, new Map(starts.map((_, index) => [`u${index}`, index])))
      const original = Element.prototype.getBoundingClientRect
      let outer = 0
      let inner = 0
      Element.prototype.getBoundingClientRect = function () {
        if (this.getAttribute('data-slot') === 'aui_message-group') {
          outer++
        } else if (this.closest('[data-slot="aui_message-group"]')) {
          inner++
        }

        return original.call(this)
      }

      const measurements: { wanted: number; actual: number; outer: number; inner: number; baselineReads: number }[] = []

      try {
        for (const wanted of [0, 300, 30, 500, 200]) {
          viewport.scrollTop = starts[wanted] + 20
          await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
          outer = inner = 0
          const actual = reader.read()
          const measured = { wanted, actual, outer, inner, baselineReads: 0 }
          outer = inner = 0
          const threshold = viewport.getBoundingClientRect().top + 8
          let baseline = 0

          for (const node of viewport.querySelectorAll<HTMLElement>('[data-message-id]')) {
            const turn = node.closest<HTMLElement>('[data-slot="aui_turn-pair"]')!

            if (turn.getBoundingClientRect().top <= threshold) {
              baseline = Number(node.dataset.messageId!.slice(1))
            }
          }

          if (baseline !== actual) {
            throw new Error(`Reference scan ${baseline} disagrees with indexed position ${actual}`)
          }

          measured.baselineReads = inner
          measurements.push(measured)
        }
      } finally {
        Element.prototype.getBoundingClientRect = original
      }

      return measurements
    })

    for (const row of results) {
      expect(row.actual).toBe(row.wanted)
      expect(row.inner).toBe(0)
      expect(row.outer).toBeLessThanOrEqual(Math.ceil(Math.log2(512)) + 1)
      expect(row.baselineReads).toBe(512)
    }

    await testInfo.attach('timeline-measurements.json', { body: JSON.stringify(results, null, 2), contentType: 'application/json' })
    await page.screenshot({ path: testInfo.outputPath('timeline-geometry.png') })
  } finally {
    await app.close()
    rmSync(temporary, { recursive: true, force: true })
  }
})
