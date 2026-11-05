import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterEach, expect, it } from 'vitest'
import { assertUnpackedComplete, missingUnpackedFiles } from './assert-unpacked-complete.mjs'

const FILES = [
  'icudtl.dat',
  'resources.pak',
  'locales/en-US.pak',
  'resources/app.asar',
  'resources/app.asar.unpacked/dist/index.html'
]
const dirs = []

async function tree(skip) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'unpacked-'))
  dirs.push(root)
  for (const rel of FILES.filter(f => f !== skip)) {
    await mkdir(path.dirname(path.join(root, rel)), { recursive: true })
    await writeFile(path.join(root, rel), 'x')
  }
  return root
}

afterEach(async () => {
  await Promise.all(dirs.splice(0).map(d => rm(d, { recursive: true, force: true })))
})

it('accepts a complete tree', async () => {
  const appOutDir = await tree()
  expect(missingUnpackedFiles(appOutDir)).toEqual([])
  expect(() => assertUnpackedComplete({ appOutDir, electronPlatformName: 'win32' })).not.toThrow()
})

it.each(FILES)('rejects a tree missing %s', async skip => {
  const appOutDir = await tree(skip)
  expect(() => assertUnpackedComplete({ appOutDir, electronPlatformName: 'win32' })).toThrow(/incomplete/)
})

it('does not judge macOS bundles', async () => {
  const appOutDir = await tree('icudtl.dat')
  expect(() => assertUnpackedComplete({ appOutDir, electronPlatformName: 'darwin' })).not.toThrow()
})
