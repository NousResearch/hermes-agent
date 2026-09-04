/**
 * Per-profile passcode hashing for the desktop lock gate (#94028).
 *
 * A UI-level privacy gate, not a security boundary: the passcode is salted and
 * stretched with PBKDF2-HMAC-SHA256 and compared in constant time, but the
 * profile directory itself stays readable by the OS user — exactly like every
 * other desktop pref. The stored record is self-describing, so the KDF
 * parameters can be raised later without a migration.
 */

export const PASSCODE_ALGO = 'pbkdf2-sha256'
export const PASSCODE_ITERATIONS = 210_000
export const PASSCODE_SALT_BYTES = 16
export const PASSCODE_HASH_BYTES = 32
/** Upper bound so a corrupted record can't turn unlock into a CPU stall. */
const MAX_ITERATIONS = 10_000_000

export interface PasscodeRecord {
  algo: typeof PASSCODE_ALGO
  iterations: number
  /** base64 — random per hash, never reused. */
  salt: string
  /** base64 — derived key. */
  hash: string
}

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

export function bytesToBase64(bytes: Uint8Array): string {
  let out = ''
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i]
    const b1 = i + 1 < bytes.length ? bytes[i + 1] : 0
    const b2 = i + 2 < bytes.length ? bytes[i + 2] : 0
    out += B64[b0 >> 2]
    out += B64[((b0 & 3) << 4) | (b1 >> 4)]
    out += i + 1 < bytes.length ? B64[((b1 & 15) << 2) | (b2 >> 6)] : '='
    out += i + 2 < bytes.length ? B64[b2 & 63] : '='
  }
  return out
}

export function base64ToBytes(b64: string): Uint8Array {
  const clean = b64.replace(/=+$/, '')
  const out = new Uint8Array(Math.floor((clean.length * 6) / 8))
  let acc = 0
  let bits = 0
  let n = 0
  for (const ch of clean) {
    const v = B64.indexOf(ch)
    if (v === -1) {
      throw new Error('Invalid base64')
    }
    acc = (acc << 6) | v
    bits += 6
    if (bits >= 8) {
      bits -= 8
      out[n++] = (acc >> bits) & 0xff
    }
  }
  return out
}

function subtle(): SubtleCrypto {
  const s = globalThis.crypto?.subtle
  if (!s) {
    throw new Error('WebCrypto is unavailable; cannot hash a profile passcode')
  }
  return s
}

async function deriveKey(passcode: string, salt: Uint8Array, iterations: number): Promise<Uint8Array> {
  const material = await subtle().importKey('raw', new TextEncoder().encode(passcode), 'PBKDF2', false, ['deriveBits'])
  const bits = await subtle().deriveBits(
    { name: 'PBKDF2', hash: 'SHA-256', salt: salt.buffer.slice(salt.byteOffset, salt.byteOffset + salt.byteLength), iterations },
    material,
    PASSCODE_HASH_BYTES * 8
  )
  return new Uint8Array(bits)
}

export async function hashPasscode(passcode: string, iterations: number = PASSCODE_ITERATIONS): Promise<PasscodeRecord> {
  const salt = new Uint8Array(PASSCODE_SALT_BYTES)
  globalThis.crypto.getRandomValues(salt)
  const hash = await deriveKey(passcode, salt, iterations)
  return { algo: PASSCODE_ALGO, iterations, salt: bytesToBase64(salt), hash: bytesToBase64(hash) }
}

/** Constant-time comparison: returns on the first length check only. */
export function timingSafeEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) {
    return false
  }
  let diff = 0
  for (let i = 0; i < a.length; i++) {
    diff |= a[i] ^ b[i]
  }
  return diff === 0
}

export async function verifyPasscode(passcode: string, record: PasscodeRecord): Promise<boolean> {
  if (
    !record ||
    record.algo !== PASSCODE_ALGO ||
    typeof record.iterations !== 'number' ||
    !Number.isInteger(record.iterations) ||
    record.iterations < 1 ||
    record.iterations > MAX_ITERATIONS
  ) {
    return false
  }
  let salt: Uint8Array
  let expected: Uint8Array
  try {
    salt = base64ToBytes(record.salt)
    expected = base64ToBytes(record.hash)
  } catch {
    return false
  }
  const actual = await deriveKey(passcode, salt, record.iterations)
  return timingSafeEqual(actual, expected)
}
