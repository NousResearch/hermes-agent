import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'
import { prepareDmgbuild } from './prepare-dmgbuild.mjs'
import { packagingTargetArch } from './prepare-packaging-tools.mjs'

test('packager selects either same-OS payload architecture but refuses foreign OS', () => {
  for (const arch of ['x64', 'arm64']) assert.equal(packagingTargetArch(`${process.platform}-${arch}`), arch)
  const foreign = process.platform === 'darwin' ? 'win32' : 'darwin'
  assert.throws(() => packagingTargetArch(`${foreign}-arm64`), /same-OS/)
  assert.throws(() => packagingTargetArch(`${process.platform}-ia32`), /same-OS/)
})

test('explicit PM supplier is copied with paired Python; missing runtime fails before publication', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'dmgbuild-prepare-'))
  try {
    const vendor = path.join(root, 'pm-tools/vendor')
    const binary = path.join(vendor, 'dmgbuild')
    const python = path.join(vendor, 'python/bin/python3')
    fs.mkdirSync(path.dirname(python), { recursive: true })
    fs.writeFileSync(binary, '#!/usr/bin/env bash\nexit 1\n')
    fs.writeFileSync(python, 'paired Python fixture')
    const out = path.join(root, 'packager')
    const copied = prepareDmgbuild({ source: root, out, cache: path.join(root, 'cache'), binary })
    assert.equal(copied, path.join(out, 'dmgbuild/dmgbuild'))
    assert.equal(fs.readFileSync(path.join(path.dirname(copied), 'python/bin/python3'), 'utf8'), 'paired Python fixture')
    fs.writeFileSync(path.join(path.dirname(copied), 'python/bin/python3'), 'build-local change')
    assert.equal(fs.readFileSync(python, 'utf8'), 'paired Python fixture')
    fs.rmSync(python)
    assert.throws(() => prepareDmgbuild({ source: root, out, cache: root, binary }), /paired Python/)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
