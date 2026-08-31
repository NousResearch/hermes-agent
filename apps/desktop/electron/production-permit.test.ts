import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import {
  canonicalizeProductionPermit,
  createProductionPermitController,
  type ProductionApprovalEnvelope,
  type SafeStoragePort,
  validateProductionPermitEnvelope
} from './production-permit'

class FakeSafeStorage implements SafeStoragePort {
  constructor(private readonly available = true) {}

  isEncryptionAvailable(): boolean {
    return this.available
  }

  encryptString(value: string): Buffer {
    return Buffer.from(`encrypted:${value}`, 'utf8')
  }

  decryptString(value: Buffer): string {
    const text = value.toString('utf8')
    if (!text.startsWith('encrypted:')) {
      throw new Error('not encrypted')
    }
    return text.slice('encrypted:'.length)
  }
}

function envelope(root: string): ProductionApprovalEnvelope {
  return {
    approval_id: 'approval:test-1',
    schema: 'recursive-agent.desktop-production-approval-request/v1',
    mission_ref: 'mission:test-1',
    target_ref: 'path:test-result',
    call: {
      tool: 'write_file',
      args: { path: path.join(root, 'result.txt'), content: 'exact content' },
      frozen_clock: null
    },
    constraints: {
      validity_ms: 300_000,
      one_use: true,
      retry_allowed: false,
      network_allowed: false,
      delegation_allowed: false,
      allowed_write_root: root,
      ambiguous_outcome: 'terminal_quarantine'
    }
  }
}

describe('production permit Electron signer', () => {
  it('fails closed without encrypted safeStorage and does not create plaintext key material', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ares-permit-'))
    const controller = createProductionPermitController(root, new FakeSafeStorage(false))

    expect(() => controller.requestSignedWitness(envelope(root))).toThrow('safeStorage unavailable')
    expect(fs.readdirSync(root)).toEqual([])
  })

  it('persists encrypted key material and signs the exact daemon payload', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ares-permit-'))
    const controller = createProductionPermitController(root, new FakeSafeStorage())
    const witness = controller.requestSignedWitness(envelope(root))
    const publicMaterial = controller.publicKeyForEnrollment()
    const publicKey = crypto.createPublicKey({
      key: Buffer.from(publicMaterial.public_key, 'base64'),
      format: 'der',
      type: 'spki'
    })
    const { signature, ...payload } = witness

    expect(signature).toHaveLength(64)
    expect(witness.key_id).toBe(publicMaterial.key_id)
    expect(publicMaterial.verifier_enrollment.schema).toBe('recursive-agent.desktop-production-public-key/v1')
    expect(Buffer.from(publicMaterial.verifier_enrollment.public_key, 'base64')).toHaveLength(32)
    expect(
      crypto.verify(
        null,
        Buffer.from(canonicalizeProductionPermit(payload)),
        publicKey,
        Buffer.from(signature)
      )
    ).toBe(true)

    const keyFile = path.join(root, 'production-permit-signing-key.enc')
    const stored = fs.readFileSync(keyFile, 'utf8')
    expect(stored.startsWith('encrypted:')).toBe(true)
    expect(stored).not.toContain('BEGIN')

    const reloaded = createProductionPermitController(root, new FakeSafeStorage())
    expect(reloaded.publicKeyForEnrollment()).toEqual(publicMaterial)
  })

  it('rejects path escape and non-production calls before signing', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ares-permit-'))
    const controller = createProductionPermitController(root, new FakeSafeStorage())
    const bad = envelope(root)
    bad.call.args.path = path.join(root, '..', 'outside.txt')

    expect(() => validateProductionPermitEnvelope(bad)).toThrow('escapes root')
    expect(() => controller.requestSignedWitness(bad)).toThrow('escapes root')
    expect(fs.readdirSync(root)).toEqual([])
  })
})
