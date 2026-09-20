import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { describe, test } from 'vitest'

const hubFrames = [
  'apps/desktop/src/app/capabilities/skills/embedded-hub-picker.tsx',
  'apps/desktop/src/app/capabilities/plugins/plugins-tab.tsx',
  'apps/desktop/src/plugins/hermes-bots/skills-hub.tsx'
]

describe('embedded Hub iframe capabilities', () => {
  test('all Hub surfaces request only clipboard-write plus popup capabilities', () => {
    for (const relativePath of hubFrames) {
      const source = fs.readFileSync(path.resolve(relativePath), 'utf8')
      assert.match(source, /allow="clipboard-write"/)
      assert.match(source, /allow-popups allow-popups-to-escape-sandbox/)
      assert.doesNotMatch(source, /allow-top-navigation/)
    }
  })
})
