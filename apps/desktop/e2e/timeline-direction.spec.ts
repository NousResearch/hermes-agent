import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'

import { build } from 'esbuild'
import { createServer } from 'vite'

import { resolveElectronBinary } from './electron-binary'
import { _electron, expect, test } from './test'

/** Native geometry complements timeline-idle's real ThreadTimeline anchor contract. */
test('timeline hit targets stay in an inline-end gutter clear of message text', async ({
  browserName: _browserName
}, testInfo) => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'hermes-timeline-direction-'))
  const main = path.join(temporary, 'main.cjs')
  writeFileSync(
    main,
    `const { app, BrowserWindow } = require('electron');
app.setPath('userData', ${JSON.stringify(path.join(temporary, 'user-data'))});
app.whenReady().then(() => {
  const window = new BrowserWindow({ width: 800, height: 500,
    webPreferences: { sandbox: true, contextIsolation: true } });
  window.loadURL('about:blank');
});`
  )
  const server = await createServer({ server: { host: '127.0.0.1', port: 0 } })
  await server.listen()
  const origin = server.resolvedUrls!.local[0]

  const bundle = await build({
    stdin: {
      contents: `import React from 'react';
import { createRoot } from 'react-dom/client';
import { flushSync } from 'react-dom';
import { TimelineRail } from './src/components/assistant-ui/thread/timeline-rail';
const root = createRoot(document.getElementById('rail'));
flushSync(() => root.render(React.createElement(TimelineRail, {
  activeIndex: 0,
  entries: Array.from({ length: 12 }, (_, i) => ({ id: 'prompt-' + i, preview: 'رسالة تجريبية ' + i })),
  loadingId: null,
  onJump: id => { document.body.dataset.lastJump = id }
})));`,
      loader: 'tsx',
      resolveDir: process.cwd()
    },
    alias: { '@': path.resolve('src') },
    bundle: true,
    // Tooltip chrome is outside the geometry seam; avoid booting app stores in this fixture.
    plugins: [
      {
        name: 'isolated-tooltip',
        setup(builder) {
          builder.onResolve({ filter: /^@\/components\/ui\/tooltip$/ }, () => ({
            path: 'tooltip',
            namespace: 'fixture'
          }))
          builder.onLoad({ filter: /.*/, namespace: 'fixture' }, () => ({
            contents: 'export const Tip = ({children}) => children',
            loader: 'js'
          }))
        }
      }
    ],
    format: 'iife',
    loader: { '.css': 'empty' },
    define: { 'process.env.NODE_ENV': '"development"' },
    write: false
  })

  const app = await _electron.launch({
    executablePath: resolveElectronBinary([path.resolve('.'), path.resolve('../..')]),
    args: [main]
  })

  try {
    const page = await app.firstWindow()
    await page.setContent(`<section data-slot="thread" style="position:relative;height:400px;width:100%;--composer-width:60rem">
      <div data-slot="aui_thread-content" style="width:100%;margin-inline:auto">
        <p id="message" style="margin:0;overflow-wrap:anywhere">رسالة تجريبية طويلة لاختبار عدم تداخل علامات التنقل مع النص. ${'Synthetic text. '.repeat(50)}</p>
      </div>
      <nav data-slot="thread-timeline" style="position:absolute;inset-inline-end:0;top:0;height:200px"><div id="rail"></div></nav>
    </section>`)
    await page.addStyleTag({ url: `${origin}src/styles.css?direct` })
    await page.addStyleTag({ url: `${origin}src/components/assistant-ui/thread/timeline.css?direct` })
    await page.addScriptTag({ content: bundle.outputFiles[0].text })

    for (const direction of ['rtl', 'ltr'] as const) {
      await page.evaluate(direction => {
        document.documentElement.dir = direction
        document.documentElement.lang = direction === 'rtl' ? 'ar' : 'en'
      }, direction)

      for (const width of [320, 768, 1280]) {
        await app.evaluate(
          ({ BrowserWindow }, width) => BrowserWindow.getAllWindows()[0].setContentSize(width, 500),
          width
        )
        const button = page.locator('[data-timeline-index="0"]')
        await expect(button).toBeVisible()
        await button.hover()

        const geometry = await page.evaluate(() => {
          const rail = document.querySelector('[data-slot="thread-timeline"]')!.getBoundingClientRect()
          const tick = document.querySelector('[data-timeline-index="0"]')!.getBoundingClientRect()
          const bar = document.querySelector('[data-slot="timeline-bar"]')!.getBoundingClientRect()
          const text = document.getElementById('message')!.getBoundingClientRect()

          return {
            railLeft: rail.left,
            railRight: rail.right,
            tickLeft: tick.left,
            tickRight: tick.right,
            barLeft: bar.left,
            barRight: bar.right,
            textLeft: text.left,
            textRight: text.right,
            width: innerWidth
          }
        })

        if (direction === 'rtl') {
          expect(geometry.railLeft).toBe(0)
          expect(geometry.tickRight, `${direction}/${width}: click targets do not cover text`).toBeLessThanOrEqual(
            geometry.textLeft
          )
          expect(geometry.barRight).toBeLessThanOrEqual(geometry.textLeft)
        } else {
          expect(geometry.railRight).toBe(geometry.width)
          expect(geometry.tickLeft, `${direction}/${width}: click targets do not cover text`).toBeGreaterThanOrEqual(
            geometry.textRight
          )
          expect(geometry.barLeft).toBeGreaterThanOrEqual(geometry.textRight)
        }

        await button.click()
        await expect(page.locator('body')).toHaveAttribute('data-last-jump', 'prompt-0')
        await page.screenshot({ path: testInfo.outputPath(`timeline-${direction}-${width}.png`) })
      }
    }

    await page.locator('[data-slot="thread-timeline"]').evaluate(rail => rail.remove())
    expect(
      await page
        .locator('[data-slot="aui_thread-content"]')
        .evaluate(content => getComputedStyle(content).paddingInlineEnd)
    ).toBe('24px')
  } finally {
    await app.close()
    await server.close()
  }
})
