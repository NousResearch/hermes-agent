import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createHudDragSession } from './hud-drag'

test('origin is cursor minus the grab offset captured at press', () => {
  const session = createHudDragSession()

  session.begin({ x: 500, y: 400 }, { x: 190, y: 80 })
  assert.deepEqual(session.origin({ x: 540, y: 410 }), { x: 230, y: 90 })
})

test('a drag onto another display does not depend on the previous window origin', () => {
  const session = createHudDragSession()

  // Primary display is 1920px wide. Grab while the bar sits near the right edge.
  session.begin({ x: 1880, y: 400 }, { x: 1570, y: 80 })

  // Cursor is now on the secondary display. Even if AppKit clamped the last
  // setBounds so the window never left 1920, the next origin is still on the
  // other monitor — not getPosition() plus a CSS-pixel delta.
  assert.deepEqual(session.origin({ x: 2100, y: 400 }), { x: 1790, y: 80 })

  session.end()
  assert.equal(session.origin({ x: 2100, y: 400 }), null)
})

test('crossing a mixed-DPI display keeps 1:1 tracking in native DIP', () => {
  const session = createHudDragSession()

  session.begin({ x: 1900, y: 400 }, { x: 1280, y: 200 })

  // Electron samples the cursor after the scale-factor transition. No renderer
  // CSS conversion is involved; the window origin moves by the same DIP delta.
  assert.deepEqual(session.origin({ x: 1940, y: 400 }), { x: 1320, y: 200 })
})

test('a move before begin is ignored so a stale pointer cannot jump the HUD', () => {
  const session = createHudDragSession()

  assert.equal(session.origin({ x: 10, y: 20 }), null)
})
