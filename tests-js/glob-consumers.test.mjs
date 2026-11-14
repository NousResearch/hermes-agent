import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { mkdtemp, rm, writeFile, mkdir } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import { test } from 'vitest'

const desktop = createRequire(resolve(import.meta.dirname, '../apps/desktop/package.json'))

function through(...packages) {
  let from = desktop
  for (const name of packages) from = createRequire(from.resolve(name))
  return from
}

test('asar packs and extracts brace, extglob and dot matches with ignore and unpack', async () => {
  const asar = await import(desktop.resolve('@electron/asar'))
  const dir = await mkdtemp(join(tmpdir(), 'hermes-asar-glob-'))
  const source = join(dir, 'source')
  const archive = join(dir, 'app.asar')
  try {
    await mkdir(join(source, 'config'), { recursive: true })
    await mkdir(join(source, 'conf'))
    for (const [name, value] of [
      ['config/keep.txt', 'keep'],
      ['conf/more.txt', 'more'],
      ['config/.hidden.txt', 'hidden'],
      ['config/addon.node', 'binary'],
      ['config/ignored.txt', 'ignored'],
    ]) await writeFile(join(source, name), value)

    await asar.createPackageWithOptions(source, archive, {
      pattern: '/@(config|conf)/{keep,more,.hidden,addon,ignored}*',
      globOptions: { ignore: [join(source, '**/ignored*')] },
      unpack: '*.node',
    })

    const files = asar.listPackage(archive)
    for (const name of ['config/keep.txt', 'conf/more.txt', 'config/.hidden.txt', 'config/addon.node']) {
      assert.ok(files.some(file => file.replaceAll('\\', '/').endsWith(name)), name)
    }
    assert.ok(!files.some(file => file.includes('ignored.txt')))
    assert.equal(asar.extractFile(archive, 'config/keep.txt').toString(), 'keep')
    assert.equal(asar.extractFile(archive, 'config/.hidden.txt').toString(), 'hidden')
    assert.equal(asar.extractFile(archive, 'config/addon.node').toString(), 'binary')
    assert.ok(existsSync(`${archive}.unpacked/config/addon.node`))
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('electron-winstaller legacy asar keeps its callback glob contract', async () => {
  const asar = through('electron-builder', 'app-builder-lib', 'electron-winstaller')('@electron/asar')
  const dir = await mkdtemp(join(tmpdir(), 'hermes-legacy-asar-glob-'))
  const source = join(dir, 'source')
  const archive = join(dir, 'app.asar')
  try {
    await mkdir(join(source, 'config'), { recursive: true })
    await writeFile(join(source, 'config', 'value.txt'), 'legacy')
    await asar.createPackageWithOptions(source, archive, { pattern: '/{config,conf}/**/*' })
    assert.equal(asar.extractFile(archive, 'config/value.txt').toString(), 'legacy')
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('cacache put/get, verify and cleanup use the installed glob', async () => {
  const cacache = through('electron-builder', 'app-builder-lib', 'make-fetch-happen')('cacache')
  const cache = await mkdtemp(join(tmpdir(), 'hermes-cacache-glob-'))
  try {
    await cacache.put(cache, 'one', 'cached value')
    assert.equal((await cacache.get(cache, 'one')).data.toString(), 'cached value')
    await cacache.verify(cache)
    await cacache.rm.all(cache)
    await assert.rejects(cacache.get(cache, 'one'), { code: 'ENOENT' })
  } finally {
    await rm(cache, { recursive: true, force: true })
  }
})

test('node-gyp 10 consumer resolves solution and makefile globs', async () => {
  const glob = through('get-windows', 'node-gyp')('glob').glob
  const dir = await mkdtemp(join(tmpdir(), 'hermes-node-gyp-glob-'))
  try {
    await mkdir(join(dir, 'build'))
    await writeFile(join(dir, 'build', 'native.sln'), '')
    await writeFile(join(dir, 'build', 'native.mk'), '')
    assert.deepEqual(await glob('build/*.sln', { cwd: dir }), ['build/native.sln'])
    assert.deepEqual(await glob('build/*.mk', { cwd: dir }), ['build/native.mk'])
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('app-builder-lib minimatch retains brace and extglob matching', () => {
  const { minimatch } = through('electron-builder', 'app-builder-lib')('minimatch')
  assert.equal(minimatch('dist/native/addon.node', 'dist/{native,assets}/@(addon|other).node'), true)
  assert.equal(minimatch('dist/other/addon.node', 'dist/{native,assets}/@(addon|other).node'), false)
})
