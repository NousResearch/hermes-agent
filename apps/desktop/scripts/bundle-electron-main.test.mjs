import assert from 'node:assert/strict'
import fs from 'node:fs'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

const script = fs.readFileSync(
  fileURLToPath(new URL('./bundle-electron-main.mjs', import.meta.url)),
  'utf8',
)

test('production main bundle is split into stable entry and hashed ESM chunks', () => {
  // A single multi-megabyte main-process file combines every privileged API
  // call into the same AV scan target. Keep electron-main.mjs as Electron's
  // stable entry point, but let esbuild place implementation chunks alongside
  // it so a package never recreates that monolithic artifact.
  assert.match(script, /entryPoints:\s*\{\s*'electron-main': mainEntry\s*\}/)
  assert.match(script, /outdir: distDir/)
  assert.match(script, /splitting: true/)
  assert.match(script, /outExtension:\s*\{\s*'\.js': '\.mjs'\s*\}/)
  assert.match(script, /chunkNames:\s*'chunks\/\[name\]-\[hash\]'/)
})
