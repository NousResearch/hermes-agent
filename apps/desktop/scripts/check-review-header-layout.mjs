// Run from apps/desktop: node scripts/check-review-header-layout.mjs --channel chrome
// Real Chromium layout regression: jsdom cannot measure clipping or text overflow.
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { chromium } from 'playwright'
import { createServer, transformWithEsbuild } from 'vite'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const argument = name => {
  const index = process.argv.indexOf(name)
  return index < 0 ? undefined : process.argv[index + 1]
}
const output = argument('--out')
const entry = `
import React from 'react'
import { createRoot } from 'react-dom/client'
import { I18nProvider } from '/src/i18n'
import { ReviewPaneContent } from '/src/app/contrib/panes'
import { $reviewIsRepo, $reviewFiles, $reviewLoading } from '/src/store/review'
import '/src/styles.css'
$reviewIsRepo.set(true)
$reviewFiles.set([])
$reviewLoading.set(false)
createRoot(document.getElementById('root')).render(
  <I18nProvider configClient={null} initialLocale="en">
    <div id="pane" data-tree-group="" data-zone-header="" style={{width:320,height:450,overflow:'hidden'}}>
      <ReviewPaneContent />
    </div>
  </I18nProvider>
)
`
const server = await createServer({
  root,
  configFile: path.join(root, 'vite.config.ts'),
  server: { host: '127.0.0.1', port: 0 },
  plugins: [
    {
      name: 'review-header-layout-test',
      resolveId(id) {
        if (id === '/__review_layout.tsx') return '\0review-layout.tsx'
      },
      async load(id) {
        if (id === '\0review-layout.tsx') {
          return (await transformWithEsbuild(entry, 'review-layout.tsx', { loader: 'tsx', jsx: 'automatic' })).code
        }
      },
      configureServer(s) {
        s.middlewares.use('/__review_layout.html', async (_req, res) => {
          res.setHeader('Content-Type', 'text/html')
          res.end(
            await s.transformIndexHtml(
              '/__review_layout.html',
              '<!doctype html><html><head></head><body><div id="root"></div><script type="module" src="/__review_layout.tsx"></script></body></html>'
            )
          )
        })
      }
    }
  ]
})
let browser
try {
  await server.listen()
  browser = await chromium.launch({ channel: argument('--channel'), headless: true })
  const page = await browser.newPage({ viewport: { width: 800, height: 500 } })
  const pageErrors = []
  page.on('pageerror', error => pageErrors.push(error.message))
  const address = server.httpServer.address()
  await page.goto(`http://127.0.0.1:${address.port}/__review_layout.html`, {
    waitUntil: 'domcontentloaded',
    timeout: 120_000
  })
  const scope = page.getByRole('button', { name: 'Uncommitted', exact: true })
  await scope.waitFor({ timeout: 120_000 })
  await page.evaluate(() => document.fonts.ready)
  const results = []
  for (const zoneHeader of [false, true]) {
    await page.locator('#pane').evaluate((el, visible) => el.toggleAttribute('data-zone-header', visible), zoneHeader)
    for (const fontSize of [16, 20]) {
      for (const width of [160, 200, 280, 320, 400, 600]) {
        await page.evaluate(
          ({ fontSize, width }) => {
            document.documentElement.style.fontSize = `${fontSize}px`
            document.getElementById('pane').style.width = `${width}px`
          },
          { fontSize, width }
        )
        const result = await scope.evaluate(button => {
          const track = button.parentElement
          const header = track.closest('[data-suppress-pane-reveal-side]')
          const h = header.getBoundingClientRect()
          const t = track.getBoundingClientRect()
          const defects = []
          if (t.top < h.top || t.bottom > h.bottom) defects.push('scope track vertically exceeds header')
          const title = header.querySelector('[data-pane-self-label]')
          if (title.getClientRects().length) {
            const label = title.querySelector('span:not([aria-hidden])')
            if (label.scrollWidth > label.clientWidth + 1) defects.push('Review title is truncated')
            if (t.top < title.getBoundingClientRect().bottom) defects.push('scope selector shares the title row')
          }
          for (const el of [header, ...header.querySelectorAll('*')]) {
            if (el instanceof HTMLElement && el.clientWidth && el.scrollWidth > el.clientWidth + 1) {
              defects.push('header content requires horizontal scrolling or overflows')
              break
            }
          }
          for (const el of track.children) {
            const b = el.getBoundingClientRect()
            if (b.left < h.left || b.right > h.right) defects.push(`${el.textContent} is outside the pane`)
            const range = document.createRange()
            range.selectNodeContents(el)
            const text = [...range.getClientRects()]
            if (text.length !== 1) defects.push(`${el.textContent} wraps`)
            if (text.some(r => r.left < b.left - 0.5 || r.right > b.right + 0.5)) {
              defects.push(`${el.textContent} escapes its button`)
            }
          }
          for (const el of header.querySelectorAll('button[aria-label]')) {
            const b = el.getBoundingClientRect()
            if (b.left < h.left || b.right > h.right || b.top < h.top || b.bottom > h.bottom) {
              defects.push(`${el.getAttribute('aria-label')} is clipped`)
            }
          }
          return { headerHeight: h.height, trackHeight: t.height, defects }
        })
        // Every scope must be visible and reachable without scrolling the strip.
        for (const name of ['Uncommitted', 'Branch', 'Last turn']) {
          const button = page.getByRole('button', { name, exact: true })
          if (
            !(await button.evaluate(el => {
              const r = el.getBoundingClientRect()
              return el.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2))
            }))
          )
            result.defects.push(`${name} is not reachable without scrolling`)
        }
        if (output && !zoneHeader && fontSize === 16 && [160, 200, 280, 320, 600].includes(width)) {
          await fs.mkdir(output, { recursive: true })
          await page.locator('#pane').screenshot({ path: path.join(output, `review-header-${width}.png`) })
        }
        results.push({ zoneHeader, width, fontSize, ...result })
      }
    }
  }
  assert.deepEqual(pageErrors, [])
  const failures = results.filter(result => result.defects.length)
  console.log(JSON.stringify({ passed: results.length - failures.length, results }, null, 2))
  assert.deepEqual(failures, [], 'Review title and all scopes must remain fully visible while resizing')
} finally {
  await browser?.close()
  await server.close()
}
