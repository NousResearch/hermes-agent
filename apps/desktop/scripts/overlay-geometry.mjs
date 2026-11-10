// Isolated Chromium layout with real components and no live-app attachment.
// Run: pnpm exec node --test scripts/overlay-geometry.mjs
import assert from 'node:assert/strict'
import { mkdtemp, readFile, readdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'

import { chromium } from '@playwright/test'
import { compile } from '@tailwindcss/node'
import { build } from 'esbuild'

const desktop = fileURLToPath(new URL('..', import.meta.url))
async function sources(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  return (
    await Promise.all(
      entries.map(entry =>
        entry.isDirectory()
          ? sources(path.join(dir, entry.name))
          : /\.(tsx?|css)$/.test(entry.name)
            ? readFile(path.join(dir, entry.name), 'utf8')
            : ''
      )
    )
  ).flat()
}

test('shared dismiss slots do not intersect interactive content in RTL or LTR', { timeout: 240000 }, async () => {
  const scratch = await mkdtemp(path.join(tmpdir(), 'overlay-geometry-'))
  const bundle = await build({
    entryPoints: [path.join(desktop, 'scripts/fixtures/overlay-geometry.tsx')],
    absWorkingDir: desktop,
    bundle: true,
    write: false,
    format: 'iife',
    platform: 'browser',
    jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"development"', 'import.meta.env': '{}', 'import.meta.hot': 'undefined' },
    alias: { '@': path.join(desktop, 'src') },
    loader: { '.css': 'empty' }
  })
  const text =
    (await sources(path.join(desktop, 'src'))).join('\n') +
    (await readFile(path.join(desktop, 'scripts/fixtures/overlay-geometry.tsx'), 'utf8'))
  const compiler = await compile(await readFile(path.join(desktop, 'src/styles.css'), 'utf8'), {
    base: path.join(desktop, 'src'),
    onDependency() {}
  })
  const css = compiler.build([...new Set(text.match(/[^\s"'`]+/g))])
  const fixture = path.join(scratch, 'fixture.html')
  await writeFile(
    fixture,
    `<!doctype html><html><head><style>${css}\n:root {--titlebar-height:36px;--conversation-text-font-size:14px;--conversation-caption-font-size:12px;--conversation-caption-line-height:1.5;--z-modal:100;--z-modal-backdrop:99} * {animation:none!important;transition:none!important}</style></head><body><div id="root"></div><script>${bundle.outputFiles[0].text.replaceAll('</script', '<\\/script')}</script></body></html>`
  )
  const all = []
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.GEOMETRY_BROWSER || chromium.executablePath(),
    args: ['--allow-file-access-from-files']
  })
  try {
    for (const width of [400, 720, 960, 1440]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 } })
      try {
        const page = await context.newPage()
        const errors = []
        page.on('pageerror', error => errors.push(error.message))
        await page.goto(`file://${fixture}`)
        await page.locator('#geometry-result').waitFor({ state: 'attached', timeout: 60000 })
        assert.deepEqual(errors, [])
        const report = JSON.parse(await page.locator('#geometry-result').textContent())
        await writeFile(path.join(scratch, `geometry-${width}.json`), JSON.stringify(report, null, 2))
        assert.deepEqual(report.failures, [])
        assert.equal(report.results.length, 10)
        all.push(...report.results)
      } finally {
        await context.close()
      }
    }
  } finally {
    await browser.close()
  }
  console.log(`Measured ${all.length} surface/direction/width cases. Evidence: ${scratch}`)
})
