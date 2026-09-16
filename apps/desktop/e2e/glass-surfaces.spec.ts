import * as fs from 'node:fs/promises'
import * as os from 'node:os'
import * as path from 'node:path'

import { expect, test } from '@playwright/test'
import { createServer, type ViteDevServer } from 'vite'

const desktop = path.resolve(import.meta.dirname, '..')
let server: ViteDevServer
let scratch: string
let url: string

test.beforeAll(async () => {
  scratch = await fs.mkdtemp(path.join(os.tmpdir(), 'hermes-glass-'))
  // Match the existing component-browser fixtures without sharing Vite's cache.
  Object.assign(globalThis, { __dirname: desktop })
  server = await createServer({
    root: desktop,
    configFile: path.join(desktop, 'vite.config.ts'),
    configLoader: 'runner',
    cacheDir: path.join(scratch, 'node_modules/.vite'),
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    optimizeDeps: { entries: ['scripts/fixtures/glass-surfaces.html'] }
  })
  await server.listen()
  url = `${server.resolvedUrls!.local[0]}scripts/fixtures/glass-surfaces.html`
})

test.afterAll(async () => {
  await server?.close()

  if (scratch) {
    await fs.rm(scratch, { recursive: true, force: true })
  }
})

test('glass tabs fade the label beneath the close button without repainting the field', async ({ page }) => {
  await page.goto(url)
  const tab = page.getByTestId('active-tab')
  const close = tab.getByRole('button', { name: 'Close', exact: true })
  await expect(tab).toBeVisible()

  for (const id of ['active-tab', 'idle-tab', 'fixed-tab']) {
    expect(await page.getByTestId(id).evaluate(el => getComputedStyle(el).backgroundColor)).toBe('rgba(0, 0, 0, 0)')
  }

  const bounds = await tab.boundingBox()
  await tab.hover()
  await expect(close).toBeVisible()
  expect(await close.evaluate(el => getComputedStyle(el).backgroundColor)).toBe('rgba(0, 0, 0, 0)')

  const mask = await tab.getByText('A long session title', { exact: false }).evaluate(el => {
    const masks = []

    for (let node: Element | null = el; node && !node.hasAttribute('data-testid'); node = node.parentElement) {
      masks.push(getComputedStyle(node).maskImage)
    }

    return masks.find(value => value !== 'none')
  })

  expect(mask).toContain('linear-gradient')
  expect(await tab.boundingBox()).toEqual(bounds)

  // Paint a continuous marker through the real label slot to test the mask's
  // pixels, not just the existence of a gradient. The close glyph is hidden so
  // its paint cannot be mistaken for label bleed-through.
  const sample = await tab.evaluate(el => {
    const content = el.querySelector<HTMLElement>('.pane-tab-content')!
    const button = el.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!
    content.style.background = '#ff0000'
    button.style.visibility = 'hidden'
    const bounds = content.getBoundingClientRect()
    const closeBounds = button.getBoundingClientRect()

    return {
      y: Math.floor(bounds.top + 2),
      solid: Math.floor(bounds.left + 2),
      fade: Math.floor(closeBounds.left - 7),
      covered: Math.floor(closeBounds.left + 3)
    }
  })

  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))
  const pixels = await page.screenshot({ omitBackground: true, path: test.info().outputPath('masked-tab.png') })

  const alpha = await page.evaluate(
    async ({ image, points }) => {
      const img = new Image()
      img.src = image
      await img.decode()
      const canvas = document.createElement('canvas')
      canvas.width = img.width
      canvas.height = img.height
      const context = canvas.getContext('2d')!
      context.drawImage(img, 0, 0)

      return [points.solid, points.fade, points.covered].map(x => context.getImageData(x, points.y, 1, 1).data[3])
    },
    { image: `data:image/png;base64,${pixels.toString('base64')}`, points: sample }
  )

  expect(alpha[0]).toBeGreaterThan(alpha[1])
  expect(alpha[1]).toBeGreaterThan(alpha[2])
  expect(alpha[2]).toBeLessThan(128)
  await tab.evaluate(el => {
    el.querySelector<HTMLElement>('.pane-tab-content')!.style.removeProperty('background')
    el.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!.style.removeProperty('visibility')
  })
  // The active underline remains on the tab itself, without a second stroke
  // on the close button increasing its opacity.
  expect(await tab.evaluate(el => getComputedStyle(el).boxShadow)).toContain('inset')
  await close.click()
  await expect(page.locator('body')).toHaveAttribute('data-closed', 'true')
  await expect(page.locator('body')).not.toHaveAttribute('data-activated', 'true')

  for (const id of ['idle-tab', 'short-tab', 'selected-tab']) {
    const sampleTab = page.getByTestId(id)
    const before = await sampleTab.boundingBox()
    await sampleTab.hover()
    const content = sampleTab.locator('.pane-tab-content')
    expect(await content.evaluate(el => getComputedStyle(el).maskImage)).toContain('linear-gradient')
    expect(await sampleTab.boundingBox()).toEqual(before)
    expect(
      await sampleTab
        .getByRole('button', { name: 'Close', exact: true })
        .evaluate(el => getComputedStyle(el).backgroundColor)
    ).toBe('rgba(0, 0, 0, 0)')
  }

  // Glass scope and tint do not affect the fade. Solid mode still honors each
  // tab's surface token instead of becoming unconditionally transparent.
  await page.evaluate(() => {
    document.documentElement.setAttribute('data-hermes-glass-scope', 'sidebar')
    document.documentElement.style.setProperty('--translucency-glass-keep', '85%')
  })
  expect(await tab.evaluate(el => getComputedStyle(el).backgroundColor)).toBe('rgba(0, 0, 0, 0)')
  await page.evaluate(() => document.documentElement.removeAttribute('data-hermes-glass'))
  expect(await tab.evaluate(el => getComputedStyle(el).backgroundColor)).not.toBe('rgba(0, 0, 0, 0)')
  await tab.hover()
  expect(await tab.locator('.pane-tab-content').evaluate(el => getComputedStyle(el).maskImage)).toContain(
    'linear-gradient'
  )
})

test('sticky prompts clip scrolling replies without an opaque backing', async ({ page }) => {
  await page.goto(url)
  const transcript = page.getByTestId('first-transcript')
  const viewport = transcript.locator('[data-slot="aui_thread-viewport"]')
  const prompt = transcript.locator('[data-slot="aui_user-message-root"]').first()
  await expect(prompt).toBeAttached()
  await viewport.evaluate(el => {
    el.scrollTop = 0
  })
  await expect.poll(() => viewport.evaluate(el => el.scrollTop)).toBe(0)
  expect(await prompt.evaluate(el => getComputedStyle(el).backgroundColor)).toBe('rgba(0, 0, 0, 0)')
  await page.evaluate(() => document.documentElement.style.setProperty('--user-bubble-keep', '0%'))

  await viewport.evaluate(el => {
    el.scrollTop = 300
  })
  await expect.poll(() => viewport.evaluate(el => el.scrollTop)).toBe(300)

  const points = await prompt.evaluate(el => {
    const rect = el.getBoundingClientRect()
    const viewport = el.closest('[data-slot="aui_thread-viewport"]')!.getBoundingClientRect()

    return {
      x: Math.floor(rect.left + rect.width / 2),
      gap: Math.floor(viewport.top + 1),
      hidden: Math.floor(rect.top + 8),
      visible: Math.ceil(rect.bottom + 5)
    }
  })

  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))
  const image = await page.screenshot({ omitBackground: true, path: test.info().outputPath('sticky-mask.png') })

  const colors = await page.evaluate(
    async ({ image, points }) => {
      const bitmap = new Image()
      bitmap.src = image
      await bitmap.decode()
      const canvas = document.createElement('canvas')
      canvas.width = bitmap.width
      canvas.height = bitmap.height
      const context = canvas.getContext('2d')!
      context.drawImage(bitmap, 0, 0)

      return [points.gap, points.hidden, points.visible].map(y =>
        Array.from(context.getImageData(points.x, y, 1, 1).data)
      )
    },
    { image: `data:image/png;base64,${image.toString('base64')}`, points }
  )

  expect(colors[0][3]).toBeLessThan(128)
  expect(colors[1][3]).toBeLessThan(128)
  expect(colors[2]).toEqual([255, 0, 0, 255])

  const clipEdge = () =>
    transcript
      .locator('[data-slot="aui_assistant-message-root"]')
      .first()
      .evaluate(el => {
        const rect = el.getBoundingClientRect()
        const clip = Number.parseFloat(getComputedStyle(el).getPropertyValue('--sticky-prompt-clip'))

        const prompt = el
          .closest('[data-slot="aui_message-group"]')!
          .querySelector('[data-slot="aui_user-message-root"]')!

        return Math.abs(rect.top + clip - prompt.getBoundingClientRect().bottom)
      })

  await expect.poll(clipEdge).toBeLessThan(1)

  const clips = await page.evaluate(async () => {
    const transcript = document.querySelector('[data-testid="first-transcript"]')!
    const reply = transcript.querySelector<HTMLElement>('[data-slot="aui_assistant-message-root"]')!
    const read = () => getComputedStyle(reply).clipPath
    const before = read()
    document.querySelector<HTMLButtonElement>('[data-testid="first-transcript-append"]')!.click()
    const samples = [read()]

    for (let i = 0; i < 3; i++) {
      await new Promise(requestAnimationFrame)
      samples.push(read())
    }

    return { before, samples }
  })

  expect(clips.before).not.toBe('none')
  expect(clips.samples.every(value => value !== 'none')).toBe(true)

  // Expanding a pinned prompt must move the clip without needing a scroll.
  await prompt.getByRole('button').click()
  await expect.poll(() => prompt.getByRole('button').evaluate(el => el.getBoundingClientRect().height)).toBe(140)
  await expect.poll(clipEdge).toBeLessThan(1)

  // A larger secondary-window titlebar offset moves both the sticky and clip.
  await viewport.evaluate(el => el.style.setProperty('--sticky-human-top', '55px'))
  await viewport.evaluate(el => {
    el.scrollTop = 301
  })
  await expect.poll(clipEdge).toBeLessThan(1)

  // Scroll back through attachments and out of the pinned state. Nothing may
  // remain clipped, and the other pane's independently pinned reply is intact.
  await viewport.evaluate(el => el.style.removeProperty('--sticky-human-top'))
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))
  await viewport.evaluate(el => {
    el.scrollTop = 0
  })
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))
  await expect.poll(() => transcript.locator('[data-sticky-prompt-clip]').count()).toBe(0)
  await expect(transcript.getByTestId('attachment').first()).toBeVisible()
  const other = page.getByTestId('second-transcript')
  expect(await other.locator('[data-slot="aui_thread-viewport"]').evaluate(el => el.scrollTop)).toBeGreaterThan(0)
  expect(await other.locator('[data-sticky-prompt-clip]').count()).toBeGreaterThan(0)

  // Enter another turn via a large jump (including virtualized history), then
  // return: stale clip styles must leave the previous turn.
  await viewport.evaluate(el => {
    el.scrollTop = 1400
  })
  await expect.poll(() => transcript.locator('[data-sticky-prompt-clip]').count()).toBeGreaterThan(0)
  await viewport.evaluate(el => {
    el.scrollTop = 0
  })
  await expect.poll(() => transcript.locator('[data-sticky-prompt-clip]').count()).toBe(0)

  // A standalone reply can still occupy the gap below a secondary titlebar
  // when the following prompt pins. It must be clipped across group boundaries.
  await page.goto(`${url}?preceding`)
  const gapViewport = page.getByTestId('first-transcript').locator('[data-slot="aui_thread-viewport"]')
  await gapViewport.locator('[data-slot="aui_user-message-root"]').first().waitFor({ state: 'attached' })
  await gapViewport.evaluate(el => {
    el.scrollTop = 0
  })
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))

  const gapScroll = await gapViewport.evaluate(el => {
    el.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
    el.style.setProperty('--sticky-human-top', '55px')
    const group = el.querySelector('[data-slot="aui_user-message-root"]')!.closest('[data-slot="aui_message-group"]')!
    const target = el.scrollTop + group.getBoundingClientRect().top - el.getBoundingClientRect().top - 45
    el.scrollTop = target

    return el.scrollTop
  })

  await expect.poll(() => gapViewport.evaluate(el => el.scrollTop)).toBe(gapScroll)
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))

  const gapClear = () =>
    gapViewport.evaluate(el => {
      const previous = el.querySelector('[data-slot="aui_assistant-message-root"]')!
      const bounds = el.getBoundingClientRect()
      const x = bounds.left + bounds.width / 2
      const hit = document.elementFromPoint(x, bounds.top + 34)

      return { hidden: !previous.contains(hit), clip: getComputedStyle(previous).clipPath }
    })

  await expect.poll(async () => (await gapClear()).hidden).toBe(true)
  expect((await gapClear()).clip).not.toBe('none')

  // Exercise the production message/edit components as well as the colored
  // clipping markers. Editing replaces the prompt with a display:contents root.
  await page.goto(`${url}?real`)
  const realViewport = page.getByTestId('first-transcript').locator('[data-slot="aui_thread-viewport"]')
  await realViewport.locator('[data-slot="aui_assistant-message-root"]').first().waitFor()
  await realViewport
    .locator('[data-slot="aui_assistant-message-root"]')
    .first()
    .evaluate(el => {
      el.style.height = '900px'
    })
  await realViewport.evaluate(el => {
    el.scrollTop = 300
  })
  const realPrompt = realViewport.locator('[data-slot="aui_user-message-root"]').first()
  await realPrompt.getByRole('button', { name: 'Edit message', exact: true }).click()
  const editor = realViewport.getByRole('textbox', { name: 'Edit message', exact: true })
  await expect(editor).toBeVisible()
  await editor.fill('Expanded editable prompt\n'.repeat(8))
  await realViewport.evaluate(el => {
    el.scrollTop = 400
  })
  await expect
    .poll(() =>
      realViewport
        .locator('[data-slot="aui_assistant-message-root"]')
        .first()
        .evaluate(el => {
          const prompt = el
            .closest('[data-slot="aui_message-group"]')!
            .querySelector('[data-slot="aui_user-message-root"]')!

          return Math.abs(
            el.getBoundingClientRect().top +
              Number.parseFloat(getComputedStyle(el).getPropertyValue('--sticky-prompt-clip')) -
              prompt.getBoundingClientRect().bottom
          )
        })
    )
    .toBeLessThan(1)
})
