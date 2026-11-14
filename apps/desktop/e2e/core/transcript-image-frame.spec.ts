/**
 * #154409: a transcript image the renderer has not decoded yet mounts in a
 * reserved 4:3 frame. A portrait image drew at full size past that frame and
 * over the text below it. A reload clears the decoded-size cache, so the
 * stored reply mounts cold. A landscape image is the control: it fits the
 * frame either way and must stay at the top, under its download button.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from '@playwright/test'

import { coreAppEnv, createCoreSandbox, launchCoreApp, send, waitForInteractive, writeProviderHome } from './harness'
import { startScriptedProvider } from './provider'

// The scripted provider keys a turn by a `U<n>-<id>` marker in the prompt.
const PROMPT = 'U1-imageframe'
const AFTER = 'A1-imageframe'

const IMAGES = { portrait: [600, 1200], landscape: [600, 300] } as const

const svg = ([width, height]: readonly [number, number]) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="100%" height="100%" fill="#087d75"/></svg>`

test('cold image replies stay inside their reserved frames', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('image-frame')

  const lines = Object.entries(IMAGES).map(([name, size]) => {
    const file = path.join(sandbox.root, `${name}.svg`)
    fs.writeFileSync(file, svg(size))

    return `MEDIA:${file}`
  })

  writeProviderHome(sandbox.hermesHome, provider.url)
  provider.script(PROMPT, [{ text: [`${lines.join('\n\n')}\n\n${AFTER}`] }])

  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))

  try {
    await waitForInteractive(app, page)
    await send(page, PROMPT)
    await expect(page.getByText(AFTER)).toBeVisible()

    await page.reload()
    await waitForInteractive(app, page)
    const imgs = page.locator('[data-slot="aui_markdown-image"] img')
    await expect(imgs).toHaveCount(lines.length)

    for (const [index, [name, [width, height]]] of Object.entries(IMAGES).entries()) {
      const img = imgs.nth(index)
      await expect(img).toHaveJSProperty('complete', true)

      const box = await img.evaluate((element: HTMLImageElement) => {
        const frame = element.closest('[data-slot="aui_markdown-image"]')!.getBoundingClientRect()

        const download = element
          .closest('[data-slot="aui_zoomable-image"]')!
          .querySelector(':scope > button:last-child')!
          .getBoundingClientRect()

        const image = element.getBoundingClientRect()

        return {
          downloadInside:
            download.top >= image.top &&
            download.bottom <= image.bottom &&
            download.left >= image.left &&
            download.right <= image.right,
          frameRatio: frame.width / frame.height,
          imageBottomGap: frame.bottom - image.bottom,
          imageRatio: image.width / image.height,
          imageTopGap: image.top - frame.top
        }
      })

      expect(box.frameRatio, `${name}: the reloaded reply mounts in the cold 4:3 frame`).toBeCloseTo(4 / 3, 1)
      expect(box.imageTopGap, `${name}: the image is top-aligned in its frame`).toBeCloseTo(0, 0)
      expect(box.imageBottomGap, `${name}: the image paints inside its frame`).toBeGreaterThanOrEqual(-1)
      expect(box.imageRatio, `${name}: the image keeps its aspect ratio`).toBeCloseTo(width / height, 2)
      expect(box.downloadInside, `${name}: the download button sits on the image`).toBe(true)
    }
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
