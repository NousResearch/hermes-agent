import assert from 'node:assert/strict'

import { test } from 'vitest'

import packageJson from '../package.json' with { type: 'json' }

test('Windows packaging leaves the staged executable unmodified', () => {
  assert.equal(packageJson.build.win.signAndEditExecutable, false)
  assert.equal(packageJson.build.afterPack, undefined)
  assert.equal(packageJson.devDependencies.rcedit, undefined)
})
