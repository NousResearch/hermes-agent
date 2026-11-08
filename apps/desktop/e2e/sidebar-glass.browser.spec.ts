import { writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

import { expect, test } from '@playwright/test'
import { createServer, type ViteDevServer } from 'vite'

// This renderer-only regression deliberately does not boot Electron or a
// gateway. The native-material bridge records IPC; layout, RO, the real tree
// renderer, Sidebar, store and CSS all run unmodified in Chromium.
test.use({
  launchOptions: { executablePath: process.env.HERMES_BROWSER_EXECUTABLE },
  viewport: { width: 900, height: 600 }
})
let server: ViteDevServer
let url: string

test.beforeAll(async () => {
  const root = fileURLToPath(new URL('..', import.meta.url))
  server = await createServer({
    root,
    configFile: fileURLToPath(new URL('../vite.config.ts', import.meta.url)),
    plugins: [
      {
        name: 'sidebar-glass-regression-fixture',
        configureServer(vite) {
          vite.middlewares.use('/__sidebar-glass-regression', async (_request, response) => {
            response.setHeader('Content-Type', 'text/html')
            response.end(
              await vite.transformIndexHtml(
                '/__sidebar-glass-regression',
                `<!doctype html><html class="dark" dir="ltr"><head>
            <link rel="stylesheet" href="/src/styles.css">
            </head><body><div id="fixture" style="display:flex;height:100vh"></div>
            <script type="module" src="/src/store/translucency-geometry.fixture.tsx"></script></body></html>`
              )
            )
          })
        }
      }
    ],
    server: { host: '127.0.0.1', port: 0, strictPort: false, open: false }
  })
  await server.listen()
  url = `${server.resolvedUrls!.local[0]}__sidebar-glass-regression`
})

test.afterAll(async () => server?.close())

test('sidebar paint follows real cold mounts, keyed moves, reflows, visibility and direction', async ({
  page
}, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.addInitScript(() => {
    const ipc: unknown[] = []
    Object.assign(window, {
      nativeGlassCalls: ipc,
      hermesDesktop: {
        glassSupported: true,
        translucencySupported: true,
        setTranslucency: (state: unknown) => ipc.push(state)
      }
    })
  })
  await page.goto(url)
  await page.waitForFunction(() => Boolean((window as any).sidebarGlass))

  const samples = await page.evaluate(async () => {
    const api = (window as any).sidebarGlass
    const samples: Array<Record<string, any>> = []

    const sample = async (label: string) => {
      await api.settle()
      const rail = document.querySelector<HTMLElement>('[data-slot="sidebar"]')
      const rect = rail?.getBoundingClientRect()

      const visible =
        rail &&
        !rail.closest('[data-pane-hidden]') &&
        getComputedStyle(rail).visibility !== 'hidden' &&
        rect!.width > 0 &&
        rect!.height > 0

      const left = visible ? Math.max(0, Math.round(rect!.left)) : 0
      const right = visible ? Math.min(innerWidth, Math.round(rect!.right)) : 0
      const paint = getComputedStyle(document.body).backgroundImage
      const stops = [...paint.matchAll(/(-?[\d.]+)px/g)].map(match => Number(match[1]))

      // Interpret both shipped painter representations. The old two-stop
      // gradient is wrong only when its effective glass band misses Sessions.
      const band =
        stops.length === 4
          ? [stops[0], stops[2]]
          : paint.startsWith('linear-gradient(to left,')
            ? [innerWidth - stops[0], innerWidth]
            : [0, stops[0]]

      samples.push({
        label,
        left,
        right,
        paint,
        stops,
        alphas: [...paint.matchAll(/color\(srgb ([^)]+)\)/g)].map(match =>
          match[1].includes('/') ? Number(match[1].split('/')[1]) : 1
        ),
        width: rect?.width ?? 0,
        railBackground: rail ? getComputedStyle(rail).backgroundColor : null,
        rootBackground: getComputedStyle(window.document.documentElement).backgroundColor,
        paintedInterval: band[0] === band[1] ? [0, 0] : band,
        direction: getComputedStyle(document.documentElement).direction,
        state: api.state(),
        book: api.book(),
        interval: ['--glass-rail-left', '--glass-rail-right'].map(name =>
          document.documentElement.style.getPropertyValue(name)
        )
      })

      return samples[samples.length - 1]
    }

    await sample('before mount')
    api.mount()
    await sample('cold mount')
    const original = document.querySelector('[data-slot="sidebar"]')
    api.mirror()
    await sample('same-size mirror')
    const retained = original === document.querySelector('[data-slot="sidebar"]') && original!.isConnected
    // No tint/scope/resize backstop has run before either cold-mount or mirror.
    api.reorder()
    await sample('non-flip inset reorder')
    api.width(200)
    await sample('rail width resize')
    // A sibling grows during sash preview without changing the store or rail
    // width. The real tree's inline flex sizing is the same preview seam.
    const split = document.querySelector('[data-tree-split]')!
    const first = split.firstElementChild as HTMLElement
    const originalBasis = first.style.flexBasis
    first.style.flexBasis = '240px'
    await sample('neighbor preview, same rail size')
    first.style.flexBasis = originalBasis

    for (const dir of ['rtl', 'ltr']) {
      document.documentElement.dir = dir
      api.reorder()
      await sample(`${dir} inset`)
      api.mirror()
      await sample(`${dir} mirror`)
    }

    api.hide(true)
    await sample('kept-alive hidden')
    api.hide(false)
    await sample('restored')
    const settingsUnchanged = JSON.stringify(samples[0].book) === JSON.stringify(api.book())
    api.scope('window')
    await sample('full window')
    api.mode('clear')
    await sample('clear mode')
    api.mode('glass')
    api.scope('sidebar')
    await sample('sidebar re-enabled')
    api.sample = sample

    return { samples, retained, settingsUnchanged, calls: (window as any).nativeGlassCalls }
  })

  // Actual browser viewport resize/restore, not a synthetic resize event.
  await page.evaluate(() => (window as any).sidebarGlass.mirror())

  for (const width of [1100, 900]) {
    await page.setViewportSize({ width, height: 600 })
    samples.samples.push(
      await page.evaluate(async label => (window as any).sidebarGlass.sample(label), `viewport ${width}`)
    )
  }

  const evidencePath = testInfo.outputPath('sidebar-glass-evidence.json')
  await writeFile(evidencePath, JSON.stringify({ browser: page.context().browser()?.version(), ...samples }, null, 2))
  await testInfo.attach('sidebar-glass-evidence', { path: evidencePath, contentType: 'application/json' })
  await page.screenshot({ path: testInfo.outputPath('sidebar-glass.png') })
  expect(errors).toEqual([])
  expect(samples.retained).toBe(true)
  expect(samples.settingsUnchanged).toBe(true)

  for (const sample of samples.samples.filter(sample => !['full window', 'clear mode'].includes(sample.label))) {
    expect(sample.paintedInterval, sample.label).toEqual([sample.left, sample.right])

    if (sample.right > sample.left) {
      expect(sample.alphas, sample.label).toEqual([1, 0.71, 0.71, 1])
    }

    expect(sample.rootBackground, sample.label).toBe('rgba(0, 0, 0, 0)')

    if (sample.railBackground !== null) {
      expect(sample.railBackground, sample.label).toBe('rgba(0, 0, 0, 0)')
    }
  }

  expect(samples.samples.find(sample => sample.label === 'cold mount')!.width).toBe(
    samples.samples.find(sample => sample.label === 'same-size mirror')!.width
  )
  expect(samples.samples.find(sample => sample.label === 'neighbor preview, same rail size')!.width).toBe(
    samples.samples.find(sample => sample.label === 'rail width resize')!.width
  )
  expect(samples.samples.find(sample => sample.label === 'neighbor preview, same rail size')!.left).not.toBe(
    samples.samples.find(sample => sample.label === 'rail width resize')!.left
  )

  for (const dir of ['rtl', 'ltr']) {
    const inset = samples.samples.find(sample => sample.label === `${dir} inset`)!
    const mirrored = samples.samples.find(sample => sample.label === `${dir} mirror`)!

    for (const sample of [inset, mirrored]) {
      expect(sample.left, sample.label).toBeGreaterThan(0)
      expect(sample.right, sample.label).toBeLessThan(900)
    }

    expect(mirrored.left, `${dir} mirror must move the rail`).not.toBe(inset.left)
  }

  const full = samples.samples.find(sample => sample.label === 'full window')!
  expect(full.paint).toBe('none')
  expect(full.interval).toEqual(['', ''])
  expect(samples.samples.find(sample => sample.label === 'clear mode')!.interval).toEqual(['', ''])
})
