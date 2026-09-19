/**
 * Regression #110121: `npm run builder` (the `pack` / `dist` tail) must not
 * depend on any devDependency bin — a workspace whose bins were not staged
 * turned every `hermes update` desktop rebuild into `cross-env: not found`,
 * which the updater then misread as a blocked Electron download.
 */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'vitest'

import { HEAP_FLAG, builderNodeOptions } from './run-electron-builder.mjs'

const require = createRequire(import.meta.url)
const desktopDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const { scripts } = require(path.join(desktopDir, 'package.json'))

test('builder script starts with node itself, not a devDependency bin', () => {
  assert.match(scripts.builder, /^node\s/)
})

test('heap flag is applied by the wrapper and never duplicated', () => {
  assert.equal(builderNodeOptions(undefined), HEAP_FLAG)
  assert.equal(builderNodeOptions('--no-warnings'), `--no-warnings ${HEAP_FLAG}`)
  assert.equal(builderNodeOptions('--max-old-space-size=4096'), '--max-old-space-size=4096')
})
