import assert from 'node:assert/strict'
import crypto from 'node:crypto'
import os from 'node:os'
import path from 'node:path'

import { test, vi } from 'vitest'

import { execText } from './backend-claim'
import { readVerifiedHostKeyFingerprint } from './managed-rollout-host-key'

vi.mock('./backend-claim', () => ({ execText: vi.fn() }))

test('host-key identity requires one accepted key from the effective SSH configuration', async () => {
  const run = vi.mocked(execText)
  const firstKey = Buffer.from('disposable-key-one').toString('base64')
  const secondKey = Buffer.from('disposable-key-two').toString('base64')
  const config = { host: 'fixture.example.test', user: 'test', port: 2200 }
  const expanded = `hostname resolved.example.test\nport 2200\nuserknownhostsfile ${path.join(process.cwd(), 'package.json')}`

  const expected = crypto.createHash('sha256').update(Buffer.from(firstKey, 'base64'))
    .digest('base64').replace(/=+$/, '').replace(/\+/g, '-').replace(/\//g, '_')

  run.mockImplementation(async (_command, args) => {
    if (args[0] === '-G') {return expanded}

    if (args[0] === '-F') {return `resolved.example.test ssh-ed25519 ${firstKey}`}
    throw new Error('unexpected-host-key-command')
  })

  assert.equal(await readVerifiedHostKeyFingerprint(config), `SHA256:${expected}`)
  assert.equal(run.mock.calls[0][1].at(-1), 'test@fixture.example.test')
  assert.ok(run.mock.calls.slice(1).every(([_command, args]) => args[0] === '-F'))

  run.mockImplementation(async (_command, args) => {
    if (args[0] === '-G') {return expanded}

    if (args[0] === '-F') {return `resolved.example.test ssh-ed25519 ${firstKey}\nresolved.example.test ssh-rsa ${secondKey}`}
    throw new Error('unexpected-host-key-command')
  })

  await assert.rejects(readVerifiedHostKeyFingerprint(config), /ambiguous \(2 fingerprints\)/)
  run.mockReset()
})

test('host-key lookup is bounded and refuses incomplete known-hosts evidence', async () => {
  const run = vi.mocked(execText)
  const config = { host: 'fixture.example.test' }
  const expanded = `userknownhostsfile ${path.join(process.cwd(), 'package.json')}`

  run.mockResolvedValueOnce('userknownhostsfile /one /two /three /four /five')
  await assert.rejects(readVerifiedHostKeyFingerprint(config), /known-hosts file limit/)
  assert.equal(run.mock.calls.length, 1)
  run.mockReset()

  run.mockImplementation(async (_command, args, options) => {
    if (args[0] === '-G') {return expanded}
    assert.equal(options?.timeout, 5_000)
    throw Object.assign(new Error('no matching host'), { code: 1 })
  })
  await assert.rejects(readVerifiedHostKeyFingerprint(config), /ambiguous \(0 fingerprints\)/)
  run.mockReset()

  run.mockImplementation(async (_command, args) => {
    if (args[0] === '-G') {return expanded}
    throw Object.assign(new Error('known-hosts file unreadable'), { code: 255 })
  })
  await assert.rejects(readVerifiedHostKeyFingerprint(config), /known-hosts file unreadable/)
  run.mockReset()

  run.mockResolvedValueOnce(`userknownhostsfile ${path.join(os.tmpdir(), 'managed-rollout-missing-known-hosts-test')}`)
  await assert.rejects(readVerifiedHostKeyFingerprint(config), /ambiguous \(0 fingerprints\)/)
  assert.equal(run.mock.calls.length, 1)
  run.mockReset()
})
