import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'

import { build } from 'esbuild'
import { createServer } from 'vite'

import { resolveElectronBinary } from './electron-binary'
import { _electron, expect, test } from './test'

/** Exercise the real primitive and generated Tailwind CSS in native Chromium. */
test('switch thumbs stay inside the track and move toward inline-end when enabled', async ({
  browserName: _browserName
}, testInfo) => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'hermes-switch-direction-'))
  const main = path.join(temporary, 'main.cjs')
  writeFileSync(
    main,
    `const { app, BrowserWindow } = require('electron');
app.setPath('userData', ${JSON.stringify(path.join(temporary, 'user-data'))});
app.whenReady().then(() => {
  const window = new BrowserWindow({ width: 800, height: 400,
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
import { Switch } from './src/components/ui/switch';
const root = createRoot(document.getElementById('root'));
window.mountSwitch = (size, checked) => flushSync(() => root.render(
  React.createElement(Switch, { size, checked, 'aria-label': 'Direction test' })
));`,
      resolveDir: process.cwd(),
      loader: 'tsx'
    },
    alias: { '@': path.resolve('src') },
    bundle: true,
    format: 'iife',
    define: { 'process.env.NODE_ENV': '"development"' },
    write: false
  })

  const app = await _electron.launch({
    executablePath: resolveElectronBinary([path.resolve('.'), path.resolve('../..')]),
    args: [main]
  })

  try {
    const page = await app.firstWindow()
    await page.setContent('<main id="root" style="padding:80px"></main>')
    await page.addStyleTag({ url: `${origin}src/styles.css?direct` })
    await page.addStyleTag({ content: '*, *::before, *::after { transition: none !important; }' })
    await page.addScriptTag({ content: bundle.outputFiles[0].text })

    for (const direction of ['ltr', 'rtl'] as const) {
      await page.evaluate(direction => {
        document.documentElement.dir = direction
      }, direction)

      for (const size of ['default', 'xs'] as const) {
        const measure = async (checked: boolean) => {
          await page.evaluate(
            ({ size, checked }) => {
              ;(window as unknown as { mountSwitch: (size: string, checked: boolean) => void }).mountSwitch(
                size,
                checked
              )
            },
            { size, checked }
          )

          return page.locator('[data-slot="switch-thumb"]').evaluate(thumb => {
            const track = thumb.parentElement!.getBoundingClientRect()
            const rect = thumb.getBoundingClientRect()

            return { left: rect.left, right: rect.right, trackLeft: track.left, trackRight: track.right }
          })
        }

        const off = await measure(false)
        const on = await measure(true)
        expect(
          (on.left - off.left) * (direction === 'rtl' ? -1 : 1),
          `${direction}/${size}: ON moves inline-end`
        ).toBeGreaterThan(0)

        for (const rect of [off, on]) {
          expect(rect.left).toBeGreaterThanOrEqual(rect.trackLeft)
          expect(rect.right).toBeLessThanOrEqual(rect.trackRight)
        }

        await expect(page.getByRole('switch')).toHaveAttribute('aria-checked', 'true')
        await page.screenshot({ path: testInfo.outputPath(`switch-${direction}-${size}.png`) })
      }
    }
  } finally {
    await app.close()
    await server.close()
  }
})
