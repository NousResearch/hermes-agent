import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'vitest'

// `npm run clean` (the `prebuild` step) runs `tsc --build <tsconfig> --clean` for every
// tsconfig in this package. tsc deletes whatever it *would* have emitted, so a config
// whose outputs land next to its sources makes the clean step delete tracked files —
// src/plugins/*/plugin.js are the plain-ESM twins of plugin.tsx and were wiped on every
// desktop rebuild (#95671). Every config must therefore emit into build/ (gitignored).

const require = createRequire(import.meta.url)
const desktopRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const tsc = require.resolve('typescript/bin/tsc')

function wouldDelete(tsconfig) {
  const out = execFileSync(process.execPath, [tsc, '--build', tsconfig, '--clean', '--dry'], {
    cwd: desktopRoot,
    encoding: 'utf8',
  })
  return out
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.startsWith('* '))
    .map(line => path.relative(desktopRoot, line.slice(2).trim()))
}

for (const tsconfig of ['tsconfig.json', 'tsconfig.electron.json', 'tsconfig.e2e.json']) {
  test(`${tsconfig}: tsc --build --clean only targets build outputs, never sources`, () => {
    const targets = wouldDelete(tsconfig)
    const outsideBuild = targets.filter(
      file => !file.startsWith(`build${path.sep}`) && !file.endsWith('.tsbuildinfo'),
    )
    assert.deepEqual(outsideBuild, [], `clean would delete tracked sources: ${outsideBuild.join(', ')}`)
  })
}
