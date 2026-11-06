import { readFileSync } from 'node:fs'

import { expect, test } from '@playwright/test'

const css = readFileSync(new URL('../../src/app/redesign/portal-artwork.css', import.meta.url), 'utf8')
  .replace('../../assets/portal-wing.svg', '/wing.svg')

const wing = readFileSync(new URL('../../src/assets/portal-wing.svg', import.meta.url), 'utf8')

for (const dark of [false, true]) {
  for (const glass of [false, true]) {
    test(`Portal wing stays faint: ${dark ? 'dark' : 'light'}, ${glass ? 'glass' : 'opaque'}`, async ({ page }) => {
      const base = dark ? 16 : 240
      await page.setViewportSize({ width: 600, height: 600 })
      await page.route('https://portal.test/**', route => route.fulfill(route.request().url().endsWith('/wing.svg')
        ? { contentType: 'image/svg+xml', body: wing }
        : { contentType: 'text/html', body: `<!doctype html>
          <html class="${dark ? 'dark' : ''}" ${glass ? 'data-hermes-glass' : ''}>
          <style>
            :root { color-scheme: ${dark ? 'dark' : 'light'}; --ui-chat-surface-background: rgb(${base} ${base} ${base}); }
            :root[data-hermes-glass] { --ui-chat-surface-background: transparent; }
            body { margin: 0; background: rgb(${base} ${base} ${base}); }
            [data-slot="composer-bounds"] { height: 100vh; }
            ${css}
          </style><div data-portal-redesign="true"><div data-slot="composer-bounds"></div></div></html>` }))
      await page.goto('https://portal.test/')

      for (const nextDark of [dark, !dark]) {
        const nextBase = nextDark ? 16 : 240
        await page.evaluate(({ nextDark, nextBase, glass }) => {
          document.documentElement.classList.toggle('dark', nextDark)
          document.documentElement.style.colorScheme = nextDark ? 'dark' : 'light'
          document.documentElement.style.setProperty('--ui-chat-surface-background', glass ? 'transparent' : `rgb(${nextBase} ${nextBase} ${nextBase})`)
          document.body.style.backgroundColor = `rgb(${nextBase} ${nextBase} ${nextBase})`
        }, { nextDark, nextBase, glass })
        const pixels = await page.screenshot()

        const delta = await page.evaluate(async ({ png, base }) => {
          const image = new Image()
          image.src = `data:image/png;base64,${png}`
          await image.decode()
          const canvas = document.createElement('canvas')
          canvas.width = image.width
          canvas.height = image.height
          const context = canvas.getContext('2d')!
          context.drawImage(image, 0, 0)
          const rgba = context.getImageData(0, 0, image.width, image.height).data
          let peak = 0

          for (let i = 0; i < rgba.length; i++) {
            if (i % 4 !== 3) {peak = Math.max(peak, Math.abs(rgba[i]! - base))}
          }

          return peak
        }, { png: pixels.toString('base64'), base: nextBase })

        expect(delta, 'The wing should remain visible').toBeGreaterThan(0)
        expect(delta, 'Glass must not expose the full-strength artwork').toBeLessThanOrEqual(10)
      }
    })
  }
}
