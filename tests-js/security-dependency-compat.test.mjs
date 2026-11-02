import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'
import { setImmediate } from 'node:timers/promises'

import { test } from 'vitest'

const require = createRequire(import.meta.url)
const asarRequire = createRequire(require.resolve('@electron/asar'))
const installerRequire = createRequire(require.resolve('electron-winstaller'))

test('aborting an ASAR glob stops recursive filesystem traversal', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-glob-abort-'))
  try {
    fs.mkdirSync(path.join(dir, 'one', 'two', 'three'), { recursive: true })
    fs.writeFileSync(path.join(dir, 'one', 'two', 'three', 'file.txt'), 'fixture')
    const { glob } = asarRequire('glob')
    const controller = new AbortController()
    const reason = new Error('cancel fixture walk')
    let reads = 0
    const trackedFs = {
      ...fs,
      readdir(...args) {
        const callback = args.pop()
        const entries = fs.readdirSync(...args)
        reads += 1
        if (reads === 1) controller.abort(reason)
        queueMicrotask(() => callback(null, entries))
      },
    }
    await assert.rejects(glob('**/*', {
      cwd: dir, signal: controller.signal, fs: trackedFs,
    }), reason)
    await setImmediate()
    assert.equal(reads, 1, 'an aborted walk must not read child directories')
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('legacy packaging consumers work with patched matchers and ASAR', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-packaging-compat-'))
  try {
    const { FileList } = require('filelist')
    const included = path.join(dir, 'one.txt')
    fs.writeFileSync(included, 'fixture')
    fs.writeFileSync(path.join(dir, 'two.js'), 'fixture')
    assert.deepEqual(new FileList().include(path.join(dir, '*.txt')).toArray(), [included])
    const legacyGlob = createRequire(require.resolve('rimraf'))('glob')
    assert.deepEqual(legacyGlob.sync(path.join(dir, '*.txt')), [included])

    // Exercise the exact ASAR module resolved by the Windows installer, including
    // the unpack-pattern path that requires a callable minimatch default in ASAR 3.
    const asar = installerRequire('@electron/asar')
    const source = path.join(dir, 'app')
    const archive = path.join(dir, 'app.asar')
    fs.mkdirSync(source)
    fs.writeFileSync(path.join(source, 'package.json'), '{"name":"fixture"}')
    fs.writeFileSync(path.join(source, 'addon.node'), 'fixture native addon')
    await asar.createPackageWithOptions(source, archive, { unpack: '*.node' })
    assert.equal(JSON.parse(asar.extractFile(archive, 'package.json')).name, 'fixture')
    assert.equal(fs.readFileSync(`${archive}.unpacked/addon.node`, 'utf8'), 'fixture native addon')
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})
