import { test, expect } from './test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('persistent terminal overlay follows the pane after split dragging', async () => {
  const page = fixture!.page

  await page.keyboard.press('Control+`')
  await page.locator('[data-terminal-slot]').waitFor({ state: 'visible', timeout: 30_000 })
  await page.locator('[data-persistent-terminal] .xterm').waitFor({ state: 'visible', timeout: 30_000 })

  const result = await page.evaluate(async () => {
    const slot = document.querySelector('[data-terminal-slot]')
    const overlay = document.querySelector('[data-persistent-terminal]')

    if (!slot || !overlay) {
      return { drift: -1, moved: 0, target: false }
    }

    const before = slot.getBoundingClientRect()
    const target = [...document.querySelectorAll<HTMLElement>('[role="separator"]')]
      .map(element => {
        const box = element.getBoundingClientRect()
        const horizontal = box.width > box.height
        const center = horizontal
          ? (box.top + box.bottom) / 2
          : (box.left + box.right) / 2
        const sides = horizontal
          ? [before.top, before.bottom]
          : [before.left, before.right]

        return {
          element,
          box,
          horizontal,
          score: Math.min(...sides.map(side => Math.abs(center - side))),
        }
      })
      .filter(item => item.box.width > 0 && item.box.height > 0)
      .sort((a, b) => a.score - b.score)[0]

    if (!target) {
      return { drift: -1, moved: 0, target: false }
    }

    const x = target.box.left + target.box.width / 2
    const y0 = target.box.top + target.box.height / 2
    const nearestSide = target.horizontal
      ? Math.abs(y0 - before.top) < Math.abs(y0 - before.bottom)
        ? 'top'
        : 'bottom'
      : Math.abs(x - before.left) < Math.abs(x - before.right)
        ? 'left'
        : 'right'
    const deltaX = nearestSide === 'left' ? -1 : nearestSide === 'right' ? 1 : 0
    const deltaY = nearestSide === 'top' ? -1 : nearestSide === 'bottom' ? 1 : 0
    let currentX = x
    let y = y0
    const pointer = {
      bubbles: true,
      cancelable: true,
      pointerId: 71,
      pointerType: 'mouse',
      isPrimary: true,
      button: 0,
      buttons: 1,
    }

    target.element.dispatchEvent(
      new PointerEvent('pointerdown', { ...pointer, clientX: x, clientY: y }),
    )

    for (let index = 0; index < 24; index += 1) {
      currentX += deltaX
      y += deltaY
      window.dispatchEvent(
        new PointerEvent('pointermove', {
          ...pointer,
          clientX: currentX,
          clientY: y,
        }),
      )
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
    }

    window.dispatchEvent(
      new PointerEvent('pointerup', {
        ...pointer,
        buttons: 0,
        clientX: currentX,
        clientY: y,
      }),
    )
    await new Promise<void>(resolve => setTimeout(resolve, 350))
    await new Promise<void>(resolve =>
      requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
    )

    const next = slot.getBoundingClientRect()
    const fixed = overlay.getBoundingClientRect()

    return {
      drift: Math.max(
        Math.abs(next.top - fixed.top),
        Math.abs(next.left - fixed.left),
        Math.abs(next.width - fixed.width),
        Math.abs(next.height - fixed.height),
      ),
      moved: Math.max(
        Math.abs(next.top - before.top),
        Math.abs(next.left - before.left),
        Math.abs(next.width - before.width),
        Math.abs(next.height - before.height),
      ),
      target: true,
    }
  })

  expect(result.target).toBe(true)
  expect(result.moved).toBeGreaterThan(10)
  expect(result.drift).toBeLessThanOrEqual(1)
})
