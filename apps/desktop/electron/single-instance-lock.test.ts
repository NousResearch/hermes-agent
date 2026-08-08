import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

import { requestSingleInstanceLockWithDiagnostic } from './single-instance-lock'

const temporaryRoots: string[] = []

afterEach(() => {
  vi.restoreAllMocks()

  for (const root of temporaryRoots.splice(0)) {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

function singletonFixture(): {
  cookiePath: string
  lockPath: string
  socketPath: string
  userData: string
} {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-singleton-lock-'))
  const userData = path.join(root, 'user-data')
  const lockPath = path.join(userData, 'SingletonLock')
  const cookiePath = path.join(userData, 'SingletonCookie')
  const socketPath = path.join(userData, 'SingletonSocket')

  temporaryRoots.push(root)
  fs.mkdirSync(userData)
  fs.symlinkSync('desktop-host-4242', lockPath)
  fs.writeFileSync(cookiePath, 'cookie-sentinel')
  fs.symlinkSync('/tmp/hermes-live-socket-sentinel', socketPath)

  return { cookiePath, lockPath, socketPath, userData }
}

test('fails closed without retrying or modifying singleton artifacts', () => {
  const { cookiePath, lockPath, socketPath, userData } = singletonFixture()
  vi.spyOn(process, 'platform', 'get').mockReturnValue('linux')
  const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined)
  let attempts = 0

  const acquired = requestSingleInstanceLockWithDiagnostic({
    getPath: name => {
      assert.equal(name, 'userData')

      return userData
    },
    requestSingleInstanceLock: () => {
      attempts += 1

      return false
    }
  })

  assert.equal(acquired, false)
  assert.equal(attempts, 1)
  assert.equal(fs.readlinkSync(lockPath), 'desktop-host-4242')
  assert.equal(fs.readFileSync(cookiePath, 'utf8'), 'cookie-sentinel')
  assert.equal(fs.readlinkSync(socketPath), '/tmp/hermes-live-socket-sentinel')
  assert.equal(errorSpy.mock.calls.length, 1)
})

test('logs an actionable diagnostic when the lock cannot be acquired', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-singleton-lock-'))
  const userData = path.join(root, 'user-data')
  const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

  vi.spyOn(process, 'platform', 'get').mockReturnValue('linux')

  temporaryRoots.push(root)
  fs.mkdirSync(userData)

  const acquired = requestSingleInstanceLockWithDiagnostic({
    getPath: () => userData,
    requestSingleInstanceLock: () => false
  })

  assert.equal(acquired, false)
  assert.equal(errorSpy.mock.calls.length, 1)

  const message = String(errorSpy.mock.calls[0]?.[0])

  assert.match(message, /single-instance lock/i)
  assert.match(message, /another Hermes Desktop instance may be running/i)
  assert.match(message, /SingletonLock/)
  assert.match(message, /SingletonCookie/)
  assert.match(message, /SingletonSocket/)
  assert.match(message, new RegExp(userData.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')))
})

test('keeps the ordinary successful acquisition path silent', () => {
  const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined)
  let attempts = 0

  const acquired = requestSingleInstanceLockWithDiagnostic({
    getPath: () => assert.fail('userData is not needed after acquiring the lock'),
    requestSingleInstanceLock: () => {
      attempts += 1

      return true
    }
  })

  assert.equal(acquired, true)
  assert.equal(attempts, 1)
  assert.equal(errorSpy.mock.calls.length, 0)
})

test('preserves failed lock behavior without a diagnostic on non-Linux platforms', () => {
  vi.spyOn(process, 'platform', 'get').mockReturnValue('darwin')
  const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

  const acquired = requestSingleInstanceLockWithDiagnostic({
    getPath: () => assert.fail('the Linux diagnostic path must not run'),
    requestSingleInstanceLock: () => false
  })

  assert.equal(acquired, false)
  assert.equal(errorSpy.mock.calls.length, 0)
})

test('preserves the failed lock decision if diagnostic logging throws', () => {
  vi.spyOn(process, 'platform', 'get').mockReturnValue('linux')
  vi.spyOn(console, 'error').mockImplementation(() => {
    throw new Error('log sink unavailable')
  })

  const acquired = requestSingleInstanceLockWithDiagnostic({
    getPath: () => '/tmp/hermes-user-data',
    requestSingleInstanceLock: () => false
  })

  assert.equal(acquired, false)
})
