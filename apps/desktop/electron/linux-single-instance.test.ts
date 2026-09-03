import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { clearStaleLinuxSingletonLock, parseLocalSingletonLockTarget } from './linux-single-instance'

function tempUserData(tag: string) {
  return fs.mkdtempSync(path.join(os.tmpdir(), `hermes-singleton-${tag}-`))
}

function writeSingletonLinks(userDataDir: string, hostname: string, pid: number) {
  fs.symlinkSync(`${hostname}-${pid}`, path.join(userDataDir, 'SingletonLock'))
  fs.symlinkSync('cookie', path.join(userDataDir, 'SingletonCookie'))
  fs.symlinkSync('/tmp/hermes-singleton/socket', path.join(userDataDir, 'SingletonSocket'))
}

test('parses only a local hostname-pid singleton target', () => {
  assert.deepEqual(parseLocalSingletonLockTarget('mini-1234', 'mini'), {
    hostname: 'mini',
    pid: 1234
  })
  assert.equal(parseLocalSingletonLockTarget('other-host-1234', 'mini'), null)
  assert.equal(parseLocalSingletonLockTarget('mini-not-a-pid', 'mini'), null)
  assert.equal(parseLocalSingletonLockTarget('mini-0', 'mini'), null)
})

test('clears symlink singleton entries for a dead local process', () => {
  const userDataDir = tempUserData('dead')
  writeSingletonLinks(userDataDir, 'mini', 4242)

  const result = clearStaleLinuxSingletonLock(userDataDir, {
    hostname: 'mini',
    isPidAlive: () => false
  })

  assert.deepEqual(result, {
    pid: 4242,
    target: 'mini-4242',
    cleared: ['SingletonLock', 'SingletonCookie', 'SingletonSocket']
  })
  assert.equal(fs.existsSync(path.join(userDataDir, 'SingletonLock')), false)
  assert.equal(fs.existsSync(path.join(userDataDir, 'SingletonCookie')), false)
  assert.equal(fs.existsSync(path.join(userDataDir, 'SingletonSocket')), false)
})

test('preserves live, foreign, and malformed singleton locks', () => {
  for (const [tag, target, alive] of [
    ['live', 'mini-4242', true],
    ['foreign', 'other-host-4242', false],
    ['malformed', 'not-a-chromium-lock', false]
  ] as const) {
    const userDataDir = tempUserData(tag)
    fs.symlinkSync(target, path.join(userDataDir, 'SingletonLock'))
    fs.symlinkSync('cookie', path.join(userDataDir, 'SingletonCookie'))

    const result = clearStaleLinuxSingletonLock(userDataDir, {
      hostname: 'mini',
      isPidAlive: () => alive
    })

    assert.equal(result, null)
    assert.equal(fs.lstatSync(path.join(userDataDir, 'SingletonLock')).isSymbolicLink(), true)
    assert.equal(fs.lstatSync(path.join(userDataDir, 'SingletonCookie')).isSymbolicLink(), true)
  }
})

test('does not remove regular singleton entries during stale recovery', () => {
  const userDataDir = tempUserData('regular')
  fs.symlinkSync('mini-4242', path.join(userDataDir, 'SingletonLock'))
  fs.writeFileSync(path.join(userDataDir, 'SingletonCookie'), 'legacy-cookie')

  const result = clearStaleLinuxSingletonLock(userDataDir, {
    hostname: 'mini',
    isPidAlive: () => false
  })

  assert.deepEqual(result?.cleared, ['SingletonLock'])
  assert.equal(fs.readFileSync(path.join(userDataDir, 'SingletonCookie'), 'utf8'), 'legacy-cookie')
})

test('does not clear a lock that changes after the PID probe', () => {
  const userDataDir = tempUserData('race')
  const lockPath = path.join(userDataDir, 'SingletonLock')
  fs.symlinkSync('mini-4242', lockPath)
  fs.symlinkSync('cookie', path.join(userDataDir, 'SingletonCookie'))

  let reads = 0

  const readlinkSync = (entryPath: Parameters<typeof fs.readlinkSync>[0]): string => {
    reads += 1

    if (reads === 2) {
      fs.unlinkSync(lockPath)
      fs.symlinkSync('mini-9999', lockPath)
    }

    return fs.readlinkSync(entryPath) as string
  }

  const result = clearStaleLinuxSingletonLock(userDataDir, {
    hostname: 'mini',
    isPidAlive: () => false,
    readlinkSync
  })

  assert.equal(result, null)
  assert.equal(fs.readlinkSync(lockPath), 'mini-9999')
  assert.equal(fs.lstatSync(path.join(userDataDir, 'SingletonCookie')).isSymbolicLink(), true)
})
