/**
 * Renderer-side view of the UI Scale preset list: the Appearance row must show
 * exactly the presets the main-process zoom module owns (imported, not
 * restated, so the advertised range can never drift from the engine), and the
 * row keeps its long-standing shape — ascending percentages, the shipped 90%
 * default, and enough headroom above 175% for low-vision users (#113104).
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { matchUiScalePreset, UI_SCALE_PRESETS } from '../../../electron/ui-scale-presets'

test('the UI Scale row advertises the presets the zoom engine owns', () => {
  assert.deepEqual(UI_SCALE_PRESETS, ['90', '100', '110', '125', '150', '175', '200', '250'])
})

test('presets ascend strictly and start below the actual-size baseline', () => {
  const percents = UI_SCALE_PRESETS.map(preset => Number(preset))

  for (let i = 1; i < percents.length; i++) {
    assert.ok(percents[i] > percents[i - 1])
  }

  assert.ok(percents[0] < 100)
  assert.equal(percents[1], 100)
})

test('the row goes past 175% for low-vision users', () => {
  const percents = UI_SCALE_PRESETS.map(preset => Number(preset))

  assert.ok(Math.max(...percents) > 175, 'the preset ceiling must not stop at 175%')
})

test('exact percent matches a preset; shortcut steps between presets match none', () => {
  assert.equal(matchUiScalePreset(150), '150')
  assert.equal(matchUiScalePreset(90), '90')
  assert.equal(matchUiScalePreset(187), null)
  assert.equal(matchUiScalePreset(0), null)
})
