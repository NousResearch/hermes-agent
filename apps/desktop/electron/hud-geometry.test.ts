import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  applyHudResetBounds,
  defaultHudBounds,
  HUD_BOTTOM_MARGIN,
  HUD_HEIGHT,
  HUD_WIDTH,
  normalizeHudResizeBounds
} from './hud-geometry'

test('defaultHudBounds restores the standard size, centered near the bottom of the work area', () => {
  const area = { x: 0, y: 25, width: 1440, height: 875 }
  const bounds = defaultHudBounds(area)

  assert.equal(bounds.width, HUD_WIDTH)
  assert.equal(bounds.height, HUD_HEIGHT)
  assert.equal(bounds.x! - area.x, area.x + area.width - (bounds.x! + bounds.width))
  assert.ok(bounds.y! + bounds.height <= area.y + area.height)
  assert.ok(bounds.y! > area.y + area.height / 2)
})

test('defaultHudBounds fits the default layout to a small work area', () => {
  assert.deepEqual(defaultHudBounds({ x: -800, y: 0, width: 480, height: 240 }), {
    x: -800,
    y: 0,
    width: 480,
    height: 240
  })
})

test('defaultHudBounds keeps the spawn fallback when no display is available', () => {
  assert.deepEqual(defaultHudBounds(), { x: undefined, y: undefined, width: HUD_WIDTH, height: HUD_HEIGHT })
})

test('applyHudResetBounds restores the resize lock and reports native failure', () => {
  let resizable = false

  const win = {
    isDestroyed: () => false,
    isResizable: () => resizable,
    setResizable: (value: boolean) => {
      resizable = value
    },
    setBounds: () => {
      throw new Error('window disappeared')
    }
  }

  assert.equal(applyHudResetBounds(win, { x: 0, y: 0, width: 620, height: 320 }), false)
  assert.equal(resizable, false)
})

test('applyHudResetBounds flips resizable on while the size changes', () => {
  let resizable = false
  const applied: Array<{ height: number; width: number }> = []

  const win = {
    isDestroyed: () => false,
    isResizable: () => resizable,
    setResizable: (value: boolean) => {
      resizable = value
    },
    setBounds: (bounds: { height: number; width: number }) => {
      assert.equal(resizable, true)
      applied.push({ width: bounds.width, height: bounds.height })
    }
  }

  assert.equal(applyHudResetBounds(win, { x: 10, y: 20, width: 620, height: 320 }), true)
  assert.equal(resizable, false)
  assert.deepEqual(applied, [{ width: 620, height: 320 }])
})

test('normalizeHudResizeBounds rounds finite geometry and clamps the HUD minimums', () => {
  assert.deepEqual(normalizeHudResizeBounds({ x: 10.4, y: -20.6, width: 100, height: 80 }), {
    x: 10,
    y: -21,
    width: 380,
    height: 160
  })
})

test('normalizeHudResizeBounds rejects incomplete or non-finite native geometry', () => {
  assert.equal(normalizeHudResizeBounds(null), null)
  assert.equal(normalizeHudResizeBounds({ x: 0, y: 0, width: Number.NaN, height: 320 }), null)
  assert.equal(normalizeHudResizeBounds({ x: 0, y: 0, width: 620 }), null)
})

test('defaultHudBounds parks the tray entry point in the bottom-right corner, on screen', () => {
  const area = { x: 0, y: 25, width: 1440, height: 875 }
  const bounds = defaultHudBounds(area, { anchor: 'bottom-right' })

  assert.equal(bounds.width, HUD_WIDTH)
  // Its right edge sits one margin from the work area's right edge…
  assert.equal(area.x + area.width - (bounds.x! + bounds.width), HUD_BOTTOM_MARGIN)
  // …and it clears the taskbar exactly like the centered default does.
  assert.equal(bounds.y! + bounds.height + HUD_BOTTOM_MARGIN, area.y + area.height)
})

test('defaultHudBounds never places the bottom-right spawn off-screen on a narrow display', () => {
  // Narrower than the bar plus its margins: the clamp keeps x at the work
  // area's left edge instead of negative, so the window stays reachable.
  const bounds = defaultHudBounds({ x: -800, y: 0, width: 480, height: 240 }, { anchor: 'bottom-right' })

  assert.equal(bounds.x, -800)
})
