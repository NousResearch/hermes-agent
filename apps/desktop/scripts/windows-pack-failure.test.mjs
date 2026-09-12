import assert from 'node:assert/strict'
import * as fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test, vi } from 'vitest'
import { peFixture } from './pe-test-fixture.mjs'

vi.mock('node:fs', async importOriginal => {
  const actual = await importOriginal()
  return { ...actual, renameSync: vi.fn(actual.renameSync) }
})
const { default: beforePack } = await import('./before-pack.mjs')
const nativeFs = await vi.importActual('node:fs')

test('a denied Windows package park must fail without deleting the live app', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-pack-denied-'))
  const output = path.join(root, 'win-unpacked')
  fs.mkdirSync(output)
  fs.writeFileSync(path.join(output, 'Hermes.exe'), peFixture())
  fs.renameSync.mockImplementation((source, target) => {
    if (source === output) throw Object.assign(new Error('fixture: parking denied'), { code: 'EPERM' })
    return nativeFs.renameSync(source, target)
  })
  try {
    await assert.rejects(beforePack({ appOutDir: output, electronPlatformName: 'win32' }), /parking denied|preserv|rollback/i)
    assert.deepEqual(fs.readFileSync(path.join(output, 'Hermes.exe')), peFixture())
  } finally {
    fs.renameSync.mockReset().mockImplementation(nativeFs.renameSync)
    fs.rmSync(root, { recursive: true, force: true })
  }
})
