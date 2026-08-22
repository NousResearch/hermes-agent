import { chromium } from 'playwright'
import { createHash } from 'node:crypto'
import { spawn } from 'node:child_process'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'

const repo = path.resolve(import.meta.dirname, '../../..')
const desktop = path.join(repo, 'apps/desktop')
const dashboardDist = path.join(repo, 'plugins/kanban/dashboard/dist')
const dashboardCss = await readFile(path.join(dashboardDist, 'style.css'))
const dashboardJs = await readFile(path.join(dashboardDist, 'index.js'))
const widths = [320, 360, 390, 430]
const output = path.resolve(import.meta.dirname, 'attention-responsive')
const mutation = process.env.KANBAN_RESPONSIVE_MUTATION === 'overflow'
const port = 4179
await mkdir(output, { recursive: true })

const server = spawn(process.execPath, [path.join(repo, 'node_modules/vite/bin/vite.js'), '--host', '127.0.0.1', '--port', String(port), '--strictPort'], {
  cwd: desktop,
  stdio: ['ignore', 'pipe', 'pipe']
})

async function waitForServer() {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/kanban-evidence.html`)
      if (response.ok) return
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  throw new Error('Vite production-surface evidence server did not start')
}

const manifest = {
  captures: [],
  generatedBy: 'Playwright mounting production Desktop source and dashboard/dist',
  mutation: mutation ? 'forced-document-min-width' : null,
  privacySafeFixture: true,
  productionEntries: {
    dashboard: 'plugins/kanban/dashboard/dist/index.js',
    desktop: 'apps/desktop/src/plugins/kanban/board.tsx'
  },
  widths
}

try {
  await waitForServer()
  const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true })

  for (const surface of ['dashboard', 'desktop']) {
    for (const width of widths) {
      const page = await browser.newPage({ viewport: { width, height: 700 }, deviceScaleFactor: 1 })
      page.on('pageerror', error => console.error(`${surface} page error:`, error.message))
      page.on('console', message => {
        if (message.type() === 'error') console.error(`${surface} console error:`, message.text())
      })
      if (surface === 'dashboard') {
        await page.route('**/dashboard-production/style.css', route => route.fulfill({ body: dashboardCss, contentType: 'text/css' }))
        await page.route('**/dashboard-production/index.js', route => route.fulfill({ body: dashboardJs, contentType: 'text/javascript' }))
      }
      const url = surface === 'desktop' ? 'kanban-evidence.html' : 'kanban-dashboard-evidence.html'
      await page.goto(`http://127.0.0.1:${port}/${url}`, { waitUntil: 'domcontentloaded' })
      await page.getByText('Privacy-safe evidence task').first().waitFor()
      if (mutation) await page.addStyleTag({ content: `html{min-width:${width + 1}px!important}` })

      const geometry = await page.evaluate(() => ({
        body: document.body.scrollWidth,
        document: document.documentElement.scrollWidth,
        viewport: window.innerWidth
      }))
      if (geometry.document > geometry.viewport || geometry.body > geometry.viewport) {
        throw new Error(`${surface} ${width}px horizontal overflow: ${JSON.stringify(geometry)}`)
      }

      const filename = `${surface}-${width}.png`
      const file = path.join(output, filename)
      await page.screenshot({ path: file, fullPage: true })
      const bytes = await readFile(file)
      manifest.captures.push({
        bytes: bytes.length,
        filename,
        horizontalOverflow: false,
        sha256: createHash('sha256').update(bytes).digest('hex'),
        surface,
        width
      })
      await page.close()
    }
  }

  await browser.close()
  await writeFile(path.join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`)
} finally {
  server.kill('SIGTERM')
}
