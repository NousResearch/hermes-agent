/**
 * native-token-store.ts
 *
 * The encrypted-at-rest persistence seam for RFC 8252 native OAuth tokens:
 * NativeTokenSet → JSON → safeStorage blob → store file, and back again on the
 * next launch.
 *
 * Kept standalone (no `import 'electron'`) so the whole restart path unit-tests
 * with the `electron` vitest project — the same pattern as native-oauth.ts.
 * main.ts owns the electron-coupled halves and injects them: the safeStorage
 * encrypt/decrypt pair and the userData store-file read/write.
 *
 * The parser direction is the load-bearing detail. What lands on disk is the
 * *normalized* camelCase NativeTokenSet, so the reload boundary is
 * parseStoredTokenSet(). Gateway `/auth/native/token` responses are snake_case
 * and stay with parseTokenResponse(); crossing the two made the decrypted set
 * throw on every launch, which surfaced as "signed out after restart" (#73271).
 */

import { type NativeTokenSet, parseStoredTokenSet } from './native-oauth'

/** One encrypted blob as written per gateway base URL. */
export interface StoredTokenSecret {
  encoding?: string
  value?: string
}

/**
 * The narrow set of side effects main.ts owns. Everything here is injected so
 * the store/load round trip can be exercised without an Electron runtime, and
 * so production keeps using safeStorage unchanged.
 */
export interface NativeTokenStoreIo {
  /**
   * Encrypt one plaintext blob. main.ts passes the strict safeStorage helper,
   * which THROWS when the OS keychain is unavailable — that must stay loud.
   */
  encrypt: (plaintext: string) => StoredTokenSecret | null
  /** Decrypt a stored payload; returns '' when it cannot be read. */
  decrypt: (secret: any) => string
  /** Raw store-file text. Throws when the file is absent — treated as empty. */
  readStoreText: () => string
  /** Persist the store-file text (main.ts writes mode 0600 under userData). */
  writeStoreText: (text: string) => void
  rememberLog?: (message: string) => void
}

/**
 * baseUrl → encrypted payload. A missing, unreadable, or hand-mangled store
 * reads as empty rather than throwing: a failed *read* falls to the next rung.
 *
 * Arrays are rejected alongside every other non-object shape: assigning
 * store[baseUrl] on an array would set a non-index property, which
 * JSON.stringify drops on the way back out — the write would look like it
 * succeeded and the tokens would be gone on the next launch.
 */
function readStore(io: NativeTokenStoreIo): Record<string, any> {
  try {
    const parsed = JSON.parse(io.readStoreText())

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

/**
 * Write (or, with `tokens === null`, drop) one gateway's token set, merging
 * into whatever other gateways are already stored.
 */
export function persistNativeTokenSet(baseUrl: string, tokens: NativeTokenSet | null, io: NativeTokenStoreIo): void {
  const store = readStore(io)

  if (tokens) {
    // Encrypt the whole set as one blob so the refresh token never lands in
    // plaintext on disk. Deliberately outside the try below: an unusable
    // keychain is an authoritative write failure and must surface to the
    // caller, not be logged away as if the tokens were saved.
    store[baseUrl] = io.encrypt(JSON.stringify(tokens))
  } else {
    delete store[baseUrl]
  }

  try {
    io.writeStoreText(JSON.stringify(store))
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)

    io.rememberLog?.(`[native-oauth] failed to persist tokens: ${detail}`)
  }
}

/**
 * Reconstruct a gateway's token set from the stored encrypted payload. Returns
 * null when nothing is stored, when the blob cannot be decrypted, or when it
 * does not parse — never a partially-populated set.
 */
export function loadNativeTokenSet(baseUrl: string, io: NativeTokenStoreIo): NativeTokenSet | null {
  const secret = readStore(io)[baseUrl]

  if (!secret) {
    return null
  }

  try {
    const plaintext = io.decrypt(secret)

    if (!plaintext) {
      // A keychain that is merely locked/unavailable right now must not cost
      // the user their refresh token — leave the entry for the next attempt.
      io.rememberLog?.(`[native-oauth] failed to decrypt stored tokens for ${baseUrl}; keeping stored entry for retry`)

      return null
    }

    // Stored blobs are normalized camelCase sets, never raw gateway responses.
    return parseStoredTokenSet(JSON.parse(plaintext))
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)

    io.rememberLog?.(`[native-oauth] failed to load stored tokens for ${baseUrl}: ${detail}`)

    return null
  }
}
