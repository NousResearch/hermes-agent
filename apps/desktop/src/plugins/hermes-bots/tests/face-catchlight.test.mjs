import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// Catchlight contrast follows the PUPIL, not the body (image14 report,
// Aug 2026): dark bodies flip the pupils to light cream (`eyeFill`), and a
// white catchlight on a cream pupil is invisible — maroon/ink/oxblood
// avatars looked like they had "no dots in their eyes". The sparkle must be
// dark on light pupils and light on dark pupils.

test('catchlight fill flips with pupil polarity (hlFill), not hardcoded white', () => {
  assert.match(pluginSource, /const hlFill = isDarkColor\(color\) \? 'rgba\(0,0,0,0\.6\)' : 'rgba\(255,255,255,0\.85\)'/)
  // Both initial catchlight circles use it.
  assert.match(pluginSource, /'data-hb-hl-l': '1', cx: 14\.8, cy: eyeY0 - 0\.7, r: 0\.65, fill: hlFill/)
  assert.match(pluginSource, /'data-hb-hl-r': '1', cx: 24, cy: eyeY0 - 0\.7, r: 0\.65, fill: hlFill/)
  // No stray hardcoded white catchlight remains on the hb-hl elements.
  assert.doesNotMatch(pluginSource, /data-hb-hl-[lr]': '1'[^}]*rgba\(255,255,255/)
})
