import { expect, test } from '@playwright/test'
import tailwindcss from '@tailwindcss/vite'
import { writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createServer, type ViteDevServer } from 'vite'

const app = fileURLToPath(new URL('../..', import.meta.url))
let server: ViteDevServer
let origin: string

test.beforeAll(async () => {
  server = await createServer({
    configFile: false,
    root: app,
    cacheDir: path.join(test.info().project.outputDir, 'image-vite-cache'),
    plugins: [tailwindcss()],
    css: { postcss: { plugins: [] } },
    resolve: {
      alias: {
        '@': path.join(app, 'src'),
        '@hermes/plugin-sdk': path.join(app, 'src/sdk/index.ts'),
        '@hermes/shared': path.join(app, '../shared/src')
      },
      dedupe: ['react', 'react-dom']
    },
    optimizeDeps: { entries: ['src/test/markdown-image-browser.tsx'] },
    server: { host: '127.0.0.1', port: 0 }
  })
  await server.listen()
  origin = server.resolvedUrls!.local[0].replace(/\/$/, '')
})
test.afterAll(async () => {
  await server?.close()
})

test('markdown images stay inside stable frames and clear following prose on cold load and remount', async ({
  page
}, info) => {
  await page.setViewportSize({ width: 1000, height: 1000 })
  await page.route('**/*', route => (route.request().url().startsWith(origin) ? route.continue() : route.abort()))
  await page.route(`${origin}/image-fixture`, route =>
    route.fulfill({
      contentType: 'text/html',
      body: '<html><body><div id="root"></div><script type="module" src="/src/test/markdown-image-browser.tsx"></script></body></html>'
    })
  )
  await page.goto(`${origin}/image-fixture`)
  await page.waitForFunction(() => 'renderImage' in window)
  const evidence = []
  for (const [name, width, height] of [
    ['square', 1536, 1536],
    ['tall', 900, 2400],
    ['small', 90, 60],
    ['wide', 1600, 600]
  ] as const) {
    const src = `${origin}/${name}.svg`
    let release!: () => void
    const held = new Promise<void>(resolve => {
      release = resolve
    })
    await page.route(src, async route => {
      await held
      await route.fulfill({
        contentType: 'image/svg+xml',
        body: `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="100%" height="100%" fill="#087d75"/><text x="10" y="40" font-size="28" fill="white">Synthetic ${name}</text></svg>`
      })
    })
    await page.evaluate(src => (window as any).renderImage(src), src)
    const frame = page.locator('[data-slot="aui_markdown-image"]')
    await expect(frame).toBeVisible()
    const pending = await frame.boundingBox()
    release()
    await page.locator('img').evaluate((img: HTMLImageElement) => img.decode())
    for (const state of ['cold', 'remount'] as const) {
      if (state === 'remount') {
        await page.evaluate(() => (window as any).unmountImage())
        await expect(frame).toHaveCount(0)
        await page.evaluate(src => (window as any).renderImage(src), src)
        await page.locator('img').evaluate((img: HTMLImageElement) => img.decode())
      }
      const boxes = await page.evaluate(() => {
        const frame = document.querySelector('[data-slot="aui_markdown-image"]')!
        const img = frame.querySelector('img')!
        const paragraph = Array.from(document.querySelectorAll('p')).find(p =>
          p.textContent?.startsWith('Following text')
        )!
        const range = document.createRange()
        range.selectNodeContents(paragraph)
        const text = range.getBoundingClientRect()
        const box = (element: Element) => element.getBoundingClientRect().toJSON()
        const paintAtText = document.elementFromPoint(text.left + 2, text.top + text.height / 2)
        return {
          frame: box(frame),
          img: box(img),
          wrapper: box(img.closest('[data-slot="aui_zoomable-image"]')!),
          button: box(img.parentElement!),
          text: text.toJSON(),
          textUncovered: !frame.contains(paintAtText),
          natural: [img.naturalWidth, img.naturalHeight],
          margin: getComputedStyle(img).margin
        }
      })
      evidence.push({ name, state, pending, ...boxes })
      await page.screenshot({ path: info.outputPath(`${name}-${state}.png`), fullPage: true })
      await writeFile(info.outputPath('geometry.json'), JSON.stringify(evidence, null, 2))
      expect.soft(boxes.img.width).toBeLessThanOrEqual(width)
      expect.soft(boxes.img.height).toBeLessThanOrEqual(height)
      expect.soft(boxes.img.width / boxes.img.height).toBeCloseTo(width / height, 2)
      expect
        .soft(boxes.img.bottom, `${name} ${state}: painted image exceeds reserved frame`)
        .toBeLessThanOrEqual(boxes.frame.bottom + 1)
      expect.soft(boxes.img.right).toBeLessThanOrEqual(boxes.frame.right + 1)
      expect
        .soft(boxes.img.bottom, `${name} ${state}: following text is overlapped`)
        .toBeLessThanOrEqual(boxes.text.top)
      expect.soft(boxes.textUncovered).toBe(true)
      if (state === 'cold') expect.soft(await frame.boundingBox()).toEqual(pending)
    }
  }
})
