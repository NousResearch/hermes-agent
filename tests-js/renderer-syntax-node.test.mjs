import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { expect, test } from 'vitest'

const checker = new URL('../apps/desktop/scripts/assert-dist-built.mjs', import.meta.url).href

test('renderer syntax checks use the running Node without PATH and preserve explicit NODE', () => {
  const root = mkdtempSync(join(tmpdir(), 'renderer check with spaces-'))
  const asset = join(root, 'assets', 'index-fixture.js')
  mkdirSync(join(root, 'assets'))
  writeFileSync(join(root, 'index.html'), '<!doctype html>')
  const env = Object.fromEntries(Object.entries(process.env)
    .filter(([key]) => !['PATH', 'NODE'].includes(key.toUpperCase())))
  const check = (node) => JSON.parse(execFileSync(process.execPath, [
    '--input-type=module', '-e',
    `import { checkDistBuilt } from ${JSON.stringify(checker)}; console.log(JSON.stringify(checkDistBuilt(process.argv[1])))`,
    root,
  ], { cwd: root, env: { ...env, PATH: '', ...(node ? { NODE: node } : {}) }, encoding: 'utf8' }))
  try {
    // Parsing must not evaluate the renderer, even when its code would throw.
    writeFileSync(asset, 'export const value = 1; throw new Error("do not execute")')
    expect(check()).toEqual({ ok: true })
    expect(check(process.execPath)).toEqual({ ok: true })
    writeFileSync(asset, 'export const = ;')
    expect(check()).toMatchObject({ ok: false, error: expect.stringContaining('not valid ES module syntax') })
    const missingNode = join(root, 'missing-node')
    expect(check(missingNode)).toMatchObject({ ok: false, error: expect.stringContaining('could not run node') })
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
}, 120_000)
