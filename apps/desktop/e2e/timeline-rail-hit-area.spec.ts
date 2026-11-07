import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'

import { build } from 'esbuild'

import { resolveElectronBinary } from './electron-binary'
import { _electron, expect, test } from './test'

/**
 * Regression for #135098: the rail floated over the viewport's scrollbar and
 * the transcript's right edge. Native Chromium hit-testing, thumb drag and wheel
 * routing against the real rail CSS and layout helpers, without a backend.
 */
test('timeline rail leaves the scrollbar, wheel and RTL line starts to the transcript', async () => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'hermes-timeline-rail-'))
  const main = path.join(temporary, 'main.cjs')
  writeFileSync(
    main,
    `const { app, BrowserWindow } = require('electron');
app.setPath('userData', ${JSON.stringify(path.join(temporary, 'user-data'))});
app.whenReady().then(() => {
  const window = new BrowserWindow({ width: 900, height: 700,
    webPreferences: { sandbox: true, contextIsolation: true, backgroundThrottling: false } });
  window.loadURL('about:blank');
});`
  )

  const bundle = await build({
    entryPoints: [path.resolve('src/components/assistant-ui/thread/timeline-layout.ts')],
    bundle: true,
    format: 'iife',
    globalName: 'TimelineLayout',
    write: false
  })

  const railCss = readFileSync(path.resolve('src/components/assistant-ui/thread/timeline.css'), 'utf8')

  const app = await _electron.launch({
    executablePath: resolveElectronBinary([path.resolve('.'), path.resolve('../..')]),
    args: ['--no-sandbox', main]
  })

  try {
    const page = await app.firstWindow()
    const ticks = 40
    // The thread's real shape: viewport and rail are siblings in a relative
    // grid; the transcript column is `px-6`; scrollbars are 0.5rem (styles.css).
    await page.setContent(`<style>
html, body { margin: 0; height: 100%; }
*::-webkit-scrollbar { width: 0.5rem; height: 0.5rem; }
*::-webkit-scrollbar-thumb { background: #888; }
${railCss}
</style>
<div style="position: relative; display: grid; height: 100%; overflow: hidden">
  <div id="viewport" style="height: 100%; overflow-x: hidden; overflow-y: auto; overscroll-behavior: contain">
    <div id="content" style="width: 100%; margin: 0 auto; padding: 0 1.5rem; box-sizing: border-box">
      ${Array.from({ length: 300 }, (_, i) => `<p dir="auto">سلام، این خط ${i} از گفتگو است</p>`).join('')}
    </div>
  </div>
  <div id="rail">
    <div class="thread-timeline-ticks" data-slot="thread-timeline-ticks">
      <div class="thread-timeline-track" style="height: ${ticks * 7}px">
        ${Array.from({ length: ticks }, (_, i) => `<button class="thread-timeline-tick" style="top: ${i * 7}px; height: 7px"><span data-slot="timeline-bar"></span></button>`).join('')}
      </div>
    </div>
  </div>
</div>`)
    await page.addScriptTag({ content: bundle.outputFiles[0].text })

    const geometry = await page.evaluate(count => {
      const api = (
        window as unknown as {
          TimelineLayout: {
            forwardRailWheel: (event: WheelEvent, ticks: HTMLElement, viewport: HTMLElement) => boolean
            syncRailGutter: (rail: HTMLElement, viewport: HTMLElement) => void
            timelineRailStyle: (count: number) => Record<string, string | number>
          }
        }
      ).TimelineLayout

      const viewport = document.getElementById('viewport')!
      const rail = document.getElementById('rail')!
      const ticksEl = rail.querySelector<HTMLElement>('[data-slot="thread-timeline-ticks"]')!
      Object.assign(rail.style, api.timelineRailStyle(count))
      api.syncRailGutter(rail, viewport)
      rail.addEventListener('wheel', event => api.forwardRailWheel(event, ticksEl, viewport), { passive: true })
      viewport.scrollTop = (viewport.scrollHeight - viewport.clientHeight) / 2

      const v = viewport.getBoundingClientRect()
      const r = rail.getBoundingClientRect()
      const content = document.getElementById('content')!

      return {
        railLeft: r.left,
        railMid: (r.top + r.bottom) / 2,
        railRight: r.right,
        scrollbarX: v.right - (viewport.offsetWidth - viewport.clientWidth) / 2,
        textRight: content.getBoundingClientRect().right - parseFloat(getComputedStyle(content).paddingRight)
      }
    }, ticks)

    const hit = (x: number, y: number) =>
      page.evaluate(
        ([px, py]) => {
          const element = document.elementFromPoint(px, py)

          return element?.id || element?.className || null
        },
        [x, y]
      )

    const scrollTop = () => page.evaluate(() => document.getElementById('viewport')!.scrollTop)

    // Thumb drags and wheels land over several frames; read once they stop.
    const settledScrollTop = () =>
      page.evaluate(
        () =>
          new Promise<number>(resolve => {
            const viewport = document.getElementById('viewport')!
            let last = viewport.scrollTop
            let still = 0

            const check = () => {
              still = viewport.scrollTop === last ? still + 1 : 0
              last = viewport.scrollTop

              if (still >= 10) {
                resolve(last)
              } else {
                requestAnimationFrame(check)
              }
            }

            requestAnimationFrame(check)
          })
      )

    // The scrollbar beside the rail belongs to the viewport, not a tick.
    expect(await hit(geometry.scrollbarX, geometry.railMid)).toBe('viewport')
    expect(geometry.railRight).toBeLessThanOrEqual(geometry.scrollbarX)
    // The first glyph of a right-aligned line is text, not a tick.
    expect(geometry.railLeft).toBeGreaterThanOrEqual(geometry.textRight)
    expect(await hit(geometry.textRight - 2, geometry.railMid)).not.toContain('thread-timeline')

    // The thumb drags along the rail's band.
    let before = await settledScrollTop()
    await page.mouse.move(geometry.scrollbarX, geometry.railMid)
    await page.mouse.down()
    await page.mouse.move(geometry.scrollbarX, geometry.railMid + 100, { steps: 5 })
    await page.mouse.up()
    expect(await settledScrollTop()).toBeGreaterThan(before + 100)

    // A wheel over the rail scrolls the transcript, both ways.
    before = await settledScrollTop()
    await page.mouse.move(geometry.railRight - 6, geometry.railMid)
    await page.mouse.wheel(0, 400)
    await expect.poll(scrollTop).toBeGreaterThan(before + 100)

    before = await settledScrollTop()
    await page.mouse.wheel(0, -400)
    await expect.poll(scrollTop).toBeLessThan(before - 100)

    // The ticks stay clickable.
    expect(await hit(geometry.railRight - 6, geometry.railMid)).toContain('thread-timeline-tick')
  } finally {
    await app.close()
    rmSync(temporary, { recursive: true, force: true })
  }
})
