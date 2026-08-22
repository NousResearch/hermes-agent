import { test } from 'node:test'
import assert from 'node:assert/strict'
import { resolve } from 'node:path'

import { resolveDesktopDir, fontCandidates, FONT_REL } from '../scripts/fix-assets.mjs'

// The font-source resolution is the part the review flagged: the in-tree build
// (HERMES_AGENT_SRC) never creates vendor/, so a vendor-only lookup always
// throws. These lock the derive-from-desktop-dir behavior for both build modes.

test('resolveDesktopDir falls back to vendor/apps/desktop with no argument', () => {
  const root = '/repo/apps/mobile/desktop-port'
  assert.equal(resolveDesktopDir(['node', 'fix-assets.mjs'], root), resolve(root, 'vendor/apps/desktop'))
})

test('resolveDesktopDir uses the passed renderer dir (in-tree build)', () => {
  const got = resolveDesktopDir(['node', 'fix-assets.mjs', '/src/apps/desktop'], '/repo/apps/mobile/desktop-port')
  assert.equal(got, resolve('/src/apps/desktop'))
})

test('fontCandidates covers local + workspace-root for the standalone vendor layout', () => {
  const desktopDir = '/repo/apps/mobile/desktop-port/vendor/apps/desktop'
  assert.deepEqual(fontCandidates(desktopDir), [
    resolve(desktopDir, 'node_modules', FONT_REL),
    resolve('/repo/apps/mobile/desktop-port/vendor/node_modules', FONT_REL),
  ])
})

test('fontCandidates resolves under the real source for the in-tree build (no vendor/)', () => {
  const candidates = fontCandidates('/src/apps/desktop')
  assert.deepEqual(candidates, [
    resolve('/src/apps/desktop/node_modules', FONT_REL),
    resolve('/src/node_modules', FONT_REL),
  ])
  // Regression guard for the reported bug: the in-tree lookup must not depend on
  // a vendor/ directory that build.sh never creates when HERMES_AGENT_SRC is set.
  assert.ok(
    !candidates.some((p) => p.split('/').includes('vendor')),
    'in-tree candidates must not reference vendor/',
  )
})
