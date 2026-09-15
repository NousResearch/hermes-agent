import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test, vi } from 'vitest'

import { createBackendOwnership, parseBackendOwnership } from './backend-ownership'
import { readBackendOwnershipFile } from './backend-ownership-file'

test.each(['EACCES', 'EIO', 'EBUSY'])(
  'read failure %s preserves the roster across ownership operations',
  async code => {
    const original = { nonce: 'existing', pid: 42, profile: 'default', startMarker: 'start-42' }
    const incoming = { ...original, nonce: 'incoming', pid: 43 }
    const contents = JSON.stringify({ backends: [original] })
    const error = Object.assign(new Error('roster read failed'), { code })

    const read = vi.spyOn(fs, 'readFileSync').mockImplementation(() => {
      throw error
    })

    const write = vi.fn()
    const quarantine = vi.fn()
    const stop = vi.fn()
    const probe = vi.fn(async () => true)

    const ownership = createBackendOwnership({
      store: { read: () => readBackendOwnershipFile('backend-ownership.json'), write, quarantine },
      matchesIdentity: probe,
      matchesParent: probe,
      stop
    })

    try {
      await assert.rejects(ownership.reapOrphans(), thrown => thrown === error)
      assert.throws(
        () => ownership.release(original),
        thrown => thrown === error
      )
      await assert.rejects(ownership.claim(incoming), thrown => thrown === error)
      assert.equal(write.mock.calls.length, 0)
      assert.equal(quarantine.mock.calls.length, 0)
      assert.equal(probe.mock.calls.length, 0)
      assert.deepEqual(stop.mock.calls, [[incoming]]) // Only the failed new claim is cleaned up.

      read.mockReturnValue(contents)
      await ownership.reapOrphans()
      assert.deepEqual(parseBackendOwnership(write.mock.calls[0][0]), [original])
    } finally {
      read.mockRestore()
    }
  }
)

test('native filesystem distinguishes a missing roster, a valid file, and a read error', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ownership-read-'))
  const file = path.join(directory, 'backend-ownership.json')

  try {
    assert.equal(readBackendOwnershipFile(file), null)
    const entry = { nonce: 'native', pid: 42, profile: 'default', startMarker: 'start-42' }

    const roster = createBackendOwnership({
      store: {
        read: () => readBackendOwnershipFile(file),
        write: contents => fs.writeFileSync(file, contents, 'utf8')
      },
      matchesIdentity: async () => true,
      matchesParent: async () => true,
      stop: () => {
        throw new Error('A live parent must retain its backend')
      }
    })

    await roster.claim(entry)
    await roster.reapOrphans()
    assert.deepEqual(parseBackendOwnership(fs.readFileSync(file, 'utf8')), [entry])
    assert.equal(readBackendOwnershipFile(file), fs.readFileSync(file, 'utf8'))

    // A directory at the configured file path produces a real read failure on Windows/Linux/macOS.
    const write = vi.fn()
    const stop = vi.fn()

    const ownership = createBackendOwnership({
      store: { read: () => readBackendOwnershipFile(directory), write },
      matchesIdentity: async () => true,
      matchesParent: async () => true,
      stop
    })

    await assert.rejects(ownership.reapOrphans())
    assert.equal(write.mock.calls.length, 0)
    assert.equal(stop.mock.calls.length, 0)
    assert.deepEqual(parseBackendOwnership(fs.readFileSync(file, 'utf8')), [entry])
    roster.release(entry)
    assert.deepEqual(parseBackendOwnership(fs.readFileSync(file, 'utf8')), [])
  } finally {
    fs.unlinkSync(file)
    fs.rmdirSync(directory)
  }
})
