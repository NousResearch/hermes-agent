import assert from 'node:assert/strict'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { after, before, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'
import { createServer } from 'vite'

const app = fileURLToPath(new URL('../', import.meta.url))
const scratch = await mkdtemp(path.join(tmpdir(), 'hermes-task-scroll-'))
const output = process.env.TASK_SCROLL_OUTPUT ?? scratch
let server
let browser
let url

before(async () => {
  await mkdir(output, { recursive: true })
  console.log(`Task-scroll evidence: ${output}`)
  // The runner loader avoids writing config bundles to shared node_modules.
  globalThis.__dirname = app
  server = await createServer({
    root: app,
    configFile: path.join(app, 'vite.config.ts'),
    configLoader: 'runner',
    cacheDir: path.join(scratch, 'node_modules/.vite'),
    server: { host: '127.0.0.1', port: 18120, strictPort: true },
    optimizeDeps: { entries: ['scripts/fixtures/tasks-scroll.html'] }
  })
  await server.listen()
  url = 'http://127.0.0.1:18120/scripts/fixtures/tasks-scroll.html'
  browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] })
})
after(async () => {
  await browser?.close()
  await server?.close()
  if (output !== scratch) await rm(scratch, { recursive: true, force: true })
})

for (const count of [14, 20]) {
  test(`${count} tasks: wheel reveals the final row above the composer`, async () => {
    const page = await browser.newPage({ viewport: { width: 1200, height: 800 }, reducedMotion: 'reduce' })
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    try {
      await page.goto(`${url}?tasks=${count}`, { timeout: 120_000 })
      const stack = page.locator('[data-slot="composer-status-stack"]')
      const last = page.getByText(`Phase ${count}:`, { exact: false })
      await last.waitFor()
      await page.evaluate(() => document.fonts.ready)
      const measure = () => stack.evaluate(el => {
        const rect = el.getBoundingClientRect()
        return { height: el.clientHeight, scrollHeight: el.scrollHeight, top: el.scrollTop,
          bounds: { top: rect.top, bottom: rect.bottom }, cardHeight: el.firstElementChild.getBoundingClientRect().height }
      })
      const before = await measure()
      await stack.hover()
      await page.mouse.wheel(0, 10000)
      // Wheel is asynchronous; wait for animation frames to deliver it, without
      // scrollIntoView/locator.click silently repairing the broken layout.
      await page.waitForTimeout(400)
      const after = await measure()
      const tail = await last.evaluate(el => {
        const r = el.getBoundingClientRect()
        return { top: r.top, bottom: r.bottom,
          hit: el.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)) }
      })
      const composer = await page.locator('[data-slot="fixture-composer"]').boundingBox()
      const receipt = { count, before, after, tail, composer, errors }
      console.log(JSON.stringify(receipt))
      await writeFile(path.join(output, `tasks-${count}.json`), JSON.stringify(receipt, null, 2))
      await page.screenshot({ path: path.join(output, `tasks-${count}.png`) })
      assert.deepEqual(errors, [])
      assert.ok(after.scrollHeight > after.height && after.top > before.top, 'Long task list must scroll')
      assert.ok(tail.top >= after.bounds.top && tail.bottom <= after.bounds.bottom, 'Final row must be fully inside viewport')
      assert.ok(tail.bottom <= composer.y && tail.hit, 'Final row must not be occluded by composer or card')
      // Collapse/expand must recover the same bounded, scrollable list.
      await page.mouse.wheel(0, -10000)
      await page.waitForTimeout(400)
      const header = stack.getByRole('button').first()
      await header.click()
      assert.ok((await measure()).height < before.height, 'Collapsed list must release its height')
      await header.click()
      await stack.hover()
      await page.mouse.wheel(0, 10000)
      await page.waitForTimeout(400)
      assert.ok((await measure()).top > 0, 'Re-expanded list must remain scrollable')
    } finally {
      await page.close()
    }
  })
}
