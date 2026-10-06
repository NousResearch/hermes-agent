import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { afterEach, test } from 'vitest'
import { markIconsToolsetCommonJS } from './prepare-packaging-tools.mjs'

/** @type {string[]} */
const roots = []
afterEach(() => roots.splice(0).forEach(root => fs.rmSync(root, { recursive: true, force: true })))

/**
 * Mirrors the packaged layout that breaks without the fix: an `icons` toolset
 * whose nearest enclosing package.json declares "type":"module" (the desktop
 * app's), so Node treats a .js script in it as a module.
 * @returns {{ out: string, icons: string }}
 */
function iconsFixture() {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), 'packager-icons-'))
  roots.push(out)
  fs.writeFileSync(path.join(out, 'package.json'), '{"type":"module"}')
  const icons = path.join(out, 'icons')
  fs.mkdirSync(icons)
  fs.writeFileSync(path.join(icons, 'icon-tool.js'),
    'const fs = require("node:fs"); process.stdout.write("commonjs ran")\n')
  return { out, icons }
}

test('icons toolset is marked CommonJS so Node can run the bundled icon-tool.js', () => {
  const { out, icons } = iconsFixture()

  // Red on the base: without the marker the script is treated as ESM, so
  // require is undefined and the run fails (the #132172 failure).
  const before = spawnSync(process.execPath, [path.join(icons, 'icon-tool.js')], { encoding: 'utf8' })
  assert.notEqual(before.status, 0)
  assert.match(before.stderr, /require is not defined|ES module/i)

  markIconsToolsetCommonJS(out)

  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(icons, 'package.json'), 'utf8')), { type: 'commonjs' })
  const after = spawnSync(process.execPath, [path.join(icons, 'icon-tool.js')], { encoding: 'utf8' })
  assert.equal(after.status, 0)
  assert.match(after.stdout, /commonjs ran/)
})

test('marking the icons toolset CommonJS is idempotent', () => {
  const { out, icons } = iconsFixture()
  markIconsToolsetCommonJS(out)
  markIconsToolsetCommonJS(out)
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(icons, 'package.json'), 'utf8')), { type: 'commonjs' })
})