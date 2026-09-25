import { webcrypto } from 'node:crypto'

import { describe, expect, it } from 'vitest'

import {
  base64ToBytes,
  bytesToBase64,
  hashPasscode,
  PASSCODE_ALGO,
  PASSCODE_ITERATIONS,
  timingSafeEqual,
  verifyPasscode,
  type PasscodeRecord
} from './profile-passcode'

// jsdom exposes a `crypto` without `subtle`; the real renderer (Chromium) and
// Node both provide full WebCrypto. Install Node's implementation for tests.
if (!globalThis.crypto?.subtle) {
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: webcrypto, writable: true })
}

describe('profile passcode hashing', () => {
  it('verifies the passcode that produced the record', async () => {
    const record = await hashPasscode('correct horse battery staple')
    expect(record.algo).toBe(PASSCODE_ALGO)
    expect(record.iterations).toBe(PASSCODE_ITERATIONS)
    expect(await verifyPasscode('correct horse battery staple', record)).toBe(true)
  })

  it('rejects a wrong passcode (mutation-bite: verify must actually compare)', async () => {
    const record = await hashPasscode('correct horse battery staple')
    expect(await verifyPasscode('wrong passcode', record)).toBe(false)
  })

  it('uses a fresh random salt per hash so equal passcodes never collide', async () => {
    const a = await hashPasscode('same-passcode')
    const b = await hashPasscode('same-passcode')
    expect(a.salt).not.toBe(b.salt)
    expect(a.hash).not.toBe(b.hash)
    expect(await verifyPasscode('same-passcode', b)).toBe(true)
  })

  it('rejects a tampered stored record', async () => {
    const record = await hashPasscode('secret')
    const tampered = { ...record, hash: bytesToBase64(new Uint8Array(32)) }
    expect(await verifyPasscode('secret', tampered)).toBe(false)
  })

  it('rejects malformed records instead of throwing', async () => {
    const record = await hashPasscode('secret')
    expect(await verifyPasscode('secret', { ...record, salt: '!!!not-base64!!!' })).toBe(false)
    expect(await verifyPasscode('secret', { ...record, algo: 'md5' })).toBe(false)
    expect(await verifyPasscode('secret', { ...record, iterations: 0 })).toBe(false)
    expect(await verifyPasscode('secret', { ...record, iterations: 10_000_001 })).toBe(false)
    expect(await verifyPasscode('secret', null as unknown as PasscodeRecord)).toBe(false)
  })

  it('round-trips base64', () => {
    const bytes = new Uint8Array([0, 1, 2, 253, 254, 255, 42])
    expect(base64ToBytes(bytesToBase64(bytes))).toEqual(bytes)
  })

  it('compares in constant-time style: mismatches and length drift are unequal', () => {
    expect(timingSafeEqual(new Uint8Array([1, 2, 3]), new Uint8Array([1, 2, 3]))).toBe(true)
    expect(timingSafeEqual(new Uint8Array([1, 2, 3]), new Uint8Array([1, 2, 4]))).toBe(false)
    expect(timingSafeEqual(new Uint8Array([1, 2]), new Uint8Array([1, 2, 3]))).toBe(false)
  })
})
