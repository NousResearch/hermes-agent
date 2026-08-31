import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

export interface ProductionApprovalEnvelope {
  approval_id: string
  schema: string
  mission_ref: string
  target_ref: string
  call: {
    tool: 'write_file'
    args: { path: string; content: string }
    frozen_clock: null
  }
  constraints: {
    validity_ms: number
    one_use: true
    retry_allowed: false
    network_allowed: false
    delegation_allowed: false
    allowed_write_root: string
    ambiguous_outcome: 'terminal_quarantine'
  }
}

export interface ProductionApprovalWitness {
  approval_id: string
  mission_ref: string
  target_ref: string
  call: ProductionApprovalEnvelope['call']
  actor: string
  effect: {
    scope_name: string
    read_roots: string[]
    write_roots: string[]
    network_allowed: false
  }
  budget: {
    max_wall_time_ms: number
    max_output_bytes: number
    max_artifact_bytes: number
  }
  policy_version: 'production-permit-v1'
  issued_at: string
  not_before: string
  expires_at: string
  retry: 'no_retry'
  delegation: 'forbidden'
  outcome_policy: 'terminal_quarantine'
  key_id: string
  signature: number[]
}

export interface SafeStoragePort {
  isEncryptionAvailable(): boolean
  encryptString(value: string): Buffer
  decryptString(value: Buffer): string
}

export interface ProductionVerifierEnrollment {
  schema: 'recursive-agent.desktop-production-public-key/v1'
  key_id: string
  /** Standard Base64 of the exact 32-byte Ed25519 verifying key. */
  public_key: string
}

export interface ProductionPermitController {
  requestSignedWitness(envelope: ProductionApprovalEnvelope): ProductionApprovalWitness
  publicKeyForEnrollment(): {
    key_id: string
    public_key: string
    public_key_sha256: string
    verifier_enrollment: ProductionVerifierEnrollment
  }
}

interface StoredKey {
  key_id: string
  private_key_pkcs8: string
  public_key_spki: string
}

const KEY_FILE = 'production-permit-signing-key.enc'
const POLICY_VERSION = 'production-permit-v1' as const
const MAX_VALIDITY_MS = 300_000
const MAX_OUTPUT_BYTES = 4_096
const MAX_ARTIFACT_BYTES = 8_192

function canonicalize(value: unknown): string {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') {
    return JSON.stringify(value)
  }

  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new Error('production permit payload contains non-finite number')
    }

    return JSON.stringify(value)
  }

  if (Array.isArray(value)) {
    return `[${value.map(canonicalize).join(',')}]`
  }

  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b))

    return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${canonicalize(item)}`).join(',')}}`
  }

  throw new Error('production permit payload contains unsupported value')
}

function sha256Hex(value: Buffer): string {
  return crypto.createHash('sha256').update(value).digest('hex')
}

function rustDateTime(value: Date): string {
  const iso = value.toISOString()

  return iso.endsWith('.000Z') ? `${iso.slice(0, -5)}Z` : iso
}

function validateEnvelope(envelope: ProductionApprovalEnvelope): void {
  if (!envelope || envelope.schema !== 'recursive-agent.desktop-production-approval-request/v1') {
    throw new Error('invalid production permit schema')
  }

  if (!envelope.approval_id || !envelope.mission_ref || !envelope.target_ref) {
    throw new Error('production permit identity required')
  }

  if (envelope.call.tool !== 'write_file' || envelope.call.frozen_clock !== null) {
    throw new Error('production permit tool denied')
  }

  if (!envelope.call.args.path || typeof envelope.call.args.content !== 'string') {
    throw new Error('production permit call denied')
  }

  const c = envelope.constraints
  if (
    c.validity_ms <= 0 ||
    c.validity_ms > MAX_VALIDITY_MS ||
    c.one_use !== true ||
    c.retry_allowed !== false ||
    c.network_allowed !== false ||
    c.delegation_allowed !== false ||
    c.ambiguous_outcome !== 'terminal_quarantine'
  ) {
    throw new Error('production permit constraints denied')
  }

  if (!path.isAbsolute(c.allowed_write_root) || !path.isAbsolute(envelope.call.args.path)) {
    throw new Error('production permit root denied')
  }

  const root = path.resolve(c.allowed_write_root)
  const target = path.resolve(envelope.call.args.path)
  if (target === root || !target.startsWith(`${root}${path.sep}`)) {
    throw new Error('production permit path escapes root')
  }
}

function signingPayload(witness: Omit<ProductionApprovalWitness, 'signature'>): unknown {
  return {
    actor: witness.actor,
    approval_id: witness.approval_id,
    call: witness.call,
    delegation: witness.delegation,
    effect: witness.effect,
    budget: witness.budget,
    expires_at: witness.expires_at,
    issued_at: witness.issued_at,
    key_id: witness.key_id,
    mission_ref: witness.mission_ref,
    not_before: witness.not_before,
    outcome_policy: witness.outcome_policy,
    policy_version: witness.policy_version,
    retry: witness.retry,
    target_ref: witness.target_ref
  }
}

export function createProductionPermitController(userDataDir: string, storage: SafeStoragePort): ProductionPermitController {
  return createProductionPermitControllerWithClock(userDataDir, storage, () => new Date())
}

export function createProductionPermitControllerWithClock(
  userDataDir: string,
  storage: SafeStoragePort,
  clock: () => Date
): ProductionPermitController {
  let cached: StoredKey | undefined

  const loadOrCreate = (): StoredKey => {
    if (cached) {
      return cached
    }

    if (!storage.isEncryptionAvailable()) {
      throw new Error('production permit safeStorage unavailable')
    }
    fs.mkdirSync(userDataDir, { recursive: true, mode: 0o700 })
    const filename = path.join(userDataDir, KEY_FILE)
    let stored: StoredKey
    if (fs.existsSync(filename)) {
      const encrypted = fs.readFileSync(filename)
      const plaintext = storage.decryptString(encrypted)
      stored = JSON.parse(plaintext) as StoredKey
      if (!stored.key_id || !stored.private_key_pkcs8 || !stored.public_key_spki) {
        throw new Error('production permit key material malformed')
      }
    } else {
      const pair = crypto.generateKeyPairSync('ed25519')
      const privateKey = pair.privateKey.export({ format: 'der', type: 'pkcs8' })
      const publicKey = pair.publicKey.export({ format: 'der', type: 'spki' })
      stored = {
        key_id: `ares-desktop-ed25519-v1:${sha256Hex(publicKey)}`,
        private_key_pkcs8: privateKey.toString('base64'),
        public_key_spki: publicKey.toString('base64')
      }
      const encrypted = storage.encryptString(JSON.stringify(stored))
      const temporary = `${filename}.tmp-${process.pid}`
      fs.writeFileSync(temporary, encrypted, { mode: 0o600, flag: 'wx' })
      fs.renameSync(temporary, filename)
      fs.chmodSync(filename, 0o600)
    }
    cached = stored
    return stored
  }

  return {
    requestSignedWitness(envelope) {
      validateEnvelope(envelope)
      const stored = loadOrCreate()
      const now = clock()
      const issuedAt = rustDateTime(now)
      const expiresAt = rustDateTime(new Date(now.getTime() + envelope.constraints.validity_ms))
      const witnessWithoutSignature: Omit<ProductionApprovalWitness, 'signature'> = {
        approval_id: envelope.approval_id,
        mission_ref: envelope.mission_ref,
        target_ref: envelope.target_ref,
        call: envelope.call,
        actor: 'ares-desktop:interactive',
        effect: {
          scope_name: `production-per-call:${envelope.approval_id}`,
          read_roots: [],
          write_roots: [envelope.constraints.allowed_write_root],
          network_allowed: false
        },
        budget: {
          max_wall_time_ms: envelope.constraints.validity_ms,
          max_output_bytes: MAX_OUTPUT_BYTES,
          max_artifact_bytes: MAX_ARTIFACT_BYTES
        },
        policy_version: POLICY_VERSION,
        issued_at: issuedAt,
        not_before: issuedAt,
        expires_at: expiresAt,
        retry: 'no_retry',
        delegation: 'forbidden',
        outcome_policy: 'terminal_quarantine',
        key_id: stored.key_id
      }
      const privateKey = crypto.createPrivateKey({ key: Buffer.from(stored.private_key_pkcs8, 'base64'), format: 'der', type: 'pkcs8' })
      const signature = crypto.sign(null, Buffer.from(canonicalize(signingPayload(witnessWithoutSignature))), privateKey)
      return { ...witnessWithoutSignature, signature: [...signature] }
    },
    publicKeyForEnrollment() {
      const stored = loadOrCreate()
      const publicKey = Buffer.from(stored.public_key_spki, 'base64')
      const jwk = crypto
        .createPublicKey({ key: publicKey, format: 'der', type: 'spki' })
        .export({ format: 'jwk' })
      if (typeof jwk.x !== 'string') {
        throw new Error('production verifier key cannot be exported as Ed25519 raw bytes')
      }
      const verifierKey = Buffer.from(jwk.x, 'base64url')
      if (verifierKey.length !== 32) {
        throw new Error('production verifier key has unexpected length')
      }
      return {
        key_id: stored.key_id,
        public_key: publicKey.toString('base64'),
        public_key_sha256: sha256Hex(publicKey),
        verifier_enrollment: {
          schema: 'recursive-agent.desktop-production-public-key/v1',
          key_id: stored.key_id,
          public_key: verifierKey.toString('base64')
        }
      }
    }
  }
}

export { canonicalize as canonicalizeProductionPermit, validateEnvelope as validateProductionPermitEnvelope }
