import { spawnSync } from 'node:child_process'
import { mkdtempSync, rmSync, symlinkSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test } from 'vitest'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')

test('build scripts run when the checkout is reached through a symlink', () => {
  const root = mkdtempSync(join(tmpdir(), 'hermes build symlink-'))
  try {
    const checkout = join(root, 'checkout')
    symlinkSync(repo, checkout, 'junction')
    const entrypoints = [
      ['scripts/build/desktop.mjs', /--source and --out are required/],
      ['scripts/build/node-deps.mjs', /--source is required/],
      ['scripts/build/electron-runtime.mjs', /--source is required/],
    ]

    for (const [script, expectedError] of entrypoints) {
      const result = spawnSync(process.execPath, [join(checkout, script)], {
        cwd: checkout,
        encoding: 'utf8',
      })
      expect(result.status, script).not.toBe(0)
      expect(result.stderr, script).toMatch(expectedError)
    }
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
