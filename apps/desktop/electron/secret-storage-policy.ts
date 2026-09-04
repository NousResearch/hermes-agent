/**
 * Single product policy for desktop secret storage.
 *
 * The secure path is the default and every malformed, missing, ambiguous, or
 * legacy policy record resolves to it. Plaintext remains available only as an
 * explicit escape hatch written by this version of the product contract.
 */

export interface SecretStoragePolicy {
  /** Keychain-backed encryption enabled; defaults on, with explicit opt-out. */
  on: boolean
  /** One-shot safeStorage-to-plaintext migration already attempted. */
  migrated: boolean
}

export const SECRET_STORAGE_POLICY_FILE = 'secure-token-storage.json'
export const SECRET_STORAGE_POLICY_VERSION = 2

export interface SecretStoragePolicyIo {
  readText: () => string
  writeText: (text: string) => void
}

interface PersistedSecretStoragePolicy extends SecretStoragePolicy {
  version: typeof SECRET_STORAGE_POLICY_VERSION
}

const SECURE_DEFAULT: SecretStoragePolicy = { on: true, migrated: false }

/**
 * Parse policy JSON without allowing JSON's last-key-wins behavior to turn a
 * duplicate member into an apparently valid policy. JSON.parse remains the
 * final syntax validator; this lexical pass only tracks object member names.
 */
function parseJsonRejectingDuplicateKeys(text: string): unknown {
  let index = 0

  const skipWhitespace = () => {
    while (/\s/.test(text[index] || '')) {
      index += 1
    }
  }

  const scanString = () => {
    const start = index
    index += 1

    while (index < text.length) {
      const character = text[index]

      if (character === '\\') {
        index += 2
        continue
      }

      index += 1

      if (character === '"') {
        return text.slice(start, index)
      }
    }

    throw new Error('Unterminated JSON string')
  }

  const scanValue = (): void => {
    skipWhitespace()
    const character = text[index]

    if (character === '"') {
      scanString()
      return
    }

    if (character === '{') {
      index += 1
      skipWhitespace()
      const keys = new Set<string>()

      if (text[index] === '}') {
        index += 1
        return
      }

      while (index < text.length) {
        skipWhitespace()

        if (text[index] !== '"') {
          throw new Error('JSON object member name must be a string')
        }

        const key = JSON.parse(scanString()) as string

        if (keys.has(key)) {
          throw new Error(`Duplicate JSON object member: ${key}`)
        }

        keys.add(key)
        skipWhitespace()

        if (text[index] !== ':') {
          throw new Error('JSON object member is missing a colon')
        }

        index += 1
        scanValue()
        skipWhitespace()

        if (text[index] === '}') {
          index += 1
          return
        }

        if (text[index] !== ',') {
          throw new Error('JSON object member is missing a comma')
        }

        index += 1
      }

      throw new Error('Unterminated JSON object')
    }

    if (character === '[') {
      index += 1
      skipWhitespace()

      if (text[index] === ']') {
        index += 1
        return
      }

      while (index < text.length) {
        scanValue()
        skipWhitespace()

        if (text[index] === ']') {
          index += 1
          return
        }

        if (text[index] !== ',') {
          throw new Error('JSON array member is missing a comma')
        }

        index += 1
      }

      throw new Error('Unterminated JSON array')
    }

    const start = index

    while (index < text.length && !/[\s,}\]]/.test(text[index])) {
      index += 1
    }

    if (start === index) {
      throw new Error('Missing JSON value')
    }
  }

  scanValue()
  skipWhitespace()

  if (index !== text.length) {
    throw new Error('Trailing JSON content')
  }

  return JSON.parse(text)
}

function isPersistedPolicy(value: unknown): value is PersistedSecretStoragePolicy {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const candidate = value as Record<string, unknown>
  const keys = Object.keys(candidate)

  return (
    keys.length === 3 &&
    keys.every(key => key === 'version' || key === 'on' || key === 'migrated') &&
    candidate.version === SECRET_STORAGE_POLICY_VERSION &&
    typeof candidate.on === 'boolean' &&
    typeof candidate.migrated === 'boolean'
  )
}

/**
 * Only a version-2 record can authorize plaintext. Two-field records were
 * written automatically by the former opt-in era, so treating them as consent
 * would preserve an ambient downgrade. They therefore migrate to secure ON.
 */
export function readSecretStoragePolicy(io: SecretStoragePolicyIo): SecretStoragePolicy {
  try {
    const parsed = parseJsonRejectingDuplicateKeys(io.readText())

    if (isPersistedPolicy(parsed)) {
      return { on: parsed.on, migrated: parsed.migrated }
    }
  } catch {
    // Missing, unreadable, malformed, duplicate-key, or stale-schema records
    // all resolve to the product's fail-closed default.
  }

  return { ...SECURE_DEFAULT }
}

export function writeSecretStoragePolicy(policy: SecretStoragePolicy, io: SecretStoragePolicyIo): void {
  const persisted: PersistedSecretStoragePolicy = {
    version: SECRET_STORAGE_POLICY_VERSION,
    on: policy.on === true,
    migrated: policy.migrated === true
  }

  io.writeText(JSON.stringify(persisted))
}

/** One stored secret blob as it appears on disk. */
interface StoredSecret {
  encoding?: string
  value?: string
}

/**
 * Decide what to do with one stored blob under the current policy.
 *
 * - keep: blob is usable as-is under this policy.
 * - migrate: safeStorage blob under explicit plaintext opt-out, before the
 *   one-shot migration has run.
 * - drop: undecryptable safeStorage blob after that migration; never touch the
 *   broken keychain again while the explicit opt-out remains active.
 */
export function classifyStoredSecret(
  secret: StoredSecret | null | undefined,
  policy: SecretStoragePolicy
): 'keep' | 'migrate' | 'drop' {
  if (!secret || typeof secret !== 'object' || secret.encoding !== 'safeStorage') {
    return 'keep'
  }

  if (policy.on) {
    return 'keep'
  }

  return policy.migrated ? 'drop' : 'migrate'
}
