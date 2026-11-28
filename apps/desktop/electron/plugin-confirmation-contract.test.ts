/**
 * The maintained contract for `ctx.confirmAction` (#158148), from the side of
 * a consuming service. `ReferenceService` is what a service does with a
 * receipt: verify the signature against the key Desktop publishes, check
 * every bound field against the proposal it issued, record the receipt
 * against the nonce BEFORE acting, and answer a redelivery with the recorded
 * outcome instead of acting again. The Desktop side is the real IPC handler.
 */
import { createPublicKey, randomBytes, verify } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterAll, describe, expect, it, vi } from 'vitest'

import {
  CONFIRMATION_CHANNEL,
  type ConfirmationReceipt,
  type ConfirmationResult,
  PUBLIC_KEY_FILE,
  RECEIPT_TYPE,
  type SignedReceipt
} from './plugin-confirmation'
import { registerPluginConfirmationIpc } from './plugin-confirmation'

const BOUND = ['actionId', 'revision', 'profile', 'serviceInstance', 'nonce'] as const

interface Proposal {
  actionId: string
  revision: string
  profile: string
  serviceInstance: string
  nonce: string
  title: string
}

type Outcome = { ok: true; applied: string } | { ok: false; reason: string }

class ReferenceService {
  readonly instance = `docsvc@${randomBytes(4).toString('hex')}`
  readonly applied: string[] = []
  private document = { id: 'doc-42', revision: 'r1' }
  private readonly issued = new Map<string, Proposal>()
  private readonly receipts = new Map<string, Outcome>()

  constructor(private readonly hermesHome: string) {}

  propose(): Proposal {
    const proposal = {
      actionId: `${this.document.id}:replace-section`,
      revision: this.document.revision,
      profile: 'default',
      serviceInstance: this.instance,
      nonce: randomBytes(16).toString('base64url'),
      title: 'Replace section 2'
    }

    this.issued.set(proposal.nonce, proposal)

    return proposal
  }

  /** Someone else edits the document while the dialog is open. */
  edit() {
    this.document = { ...this.document, revision: `r${Number(this.document.revision.slice(1)) + 1}` }
  }

  deliver(receipt: SignedReceipt): Outcome {
    const claims = this.verify(receipt)

    if (!claims) {
      return { ok: false, reason: 'bad-signature' }
    }

    const recorded = this.receipts.get(claims.nonce)

    if (recorded) {
      return recorded
    }

    const proposal = this.issued.get(claims.nonce)

    if (!proposal || BOUND.some(field => claims[field] !== proposal[field])) {
      return { ok: false, reason: 'not-issued' }
    }

    // The user approved revision N; applying it to N+1 is exactly the bug.
    const outcome: Outcome =
      claims.revision === this.document.revision
        ? { ok: true, applied: `${claims.actionId}@${claims.revision}` }
        : { ok: false, reason: 'revision-changed' }

    // Recorded before acting, so a crash after this line never replays.
    this.receipts.set(claims.nonce, outcome)
    this.issued.delete(claims.nonce)

    if (outcome.ok) {
      this.applied.push(outcome.applied)
    }

    return outcome
  }

  private verify(receipt: SignedReceipt): ConfirmationReceipt | null {
    const key = JSON.parse(fs.readFileSync(path.join(this.hermesHome, PUBLIC_KEY_FILE), 'utf8'))

    if (receipt.keyId !== key.keyId) {
      return null
    }

    const signed = verify(
      null,
      Buffer.from(receipt.payload, 'utf8'),
      createPublicKey(key.publicKey),
      Buffer.from(receipt.signature, 'base64')
    )

    if (!signed) {
      return null
    }

    const claims = JSON.parse(receipt.payload) as ConfirmationReceipt

    return claims.type === RECEIPT_TYPE && claims.decision === 'approved' ? claims : null
  }
}

const hermesHome = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-confirm-contract-')))
const handlers = new Map<string, (...args: unknown[]) => Promise<ConfirmationResult>>()
const showMessageBox = vi.fn()

registerPluginConfirmationIpc({
  ipcMain: { handle: (channel: string, handler: never) => handlers.set(channel, handler) } as never,
  dialog: { showMessageBox } as never,
  windowFor: () => ({ id: 1 }) as never,
  hermesHome
})

/** The user clicks `button` in the native dialog Desktop shows for `proposal`. */
async function confirmInDesktop(proposal: Proposal, button: 'Cancel' | 'Confirm', beforeClick = () => {}) {
  showMessageBox.mockImplementationOnce(async () => {
    beforeClick()

    return { response: button === 'Confirm' ? 1 : 0 }
  })

  return handlers.get(CONFIRMATION_CHANNEL)!({ sender: {} }, 'doc-sync', proposal)
}

const approved = (result: ConfirmationResult) => {
  expect(result.status).toBe('approved')

  return (result as Extract<ConfirmationResult, { status: 'approved' }>).receipt
}

afterAll(() => fs.rmSync(hermesHome, { recursive: true, force: true }))

describe('confirmed action contract', () => {
  it('one accepted confirmation applies the action exactly once', async () => {
    const service = new ReferenceService(hermesHome)
    const receipt = approved(await confirmInDesktop(service.propose(), 'Confirm'))

    expect(service.deliver(receipt)).toEqual({ ok: true, applied: 'doc-42:replace-section@r1' })
    expect(service.applied).toEqual(['doc-42:replace-section@r1'])
  })

  it('a revision that changed while the dialog was open is rejected and nothing is applied', async () => {
    const service = new ReferenceService(hermesHome)
    const receipt = approved(await confirmInDesktop(service.propose(), 'Confirm', () => service.edit()))

    expect(service.deliver(receipt)).toEqual({ ok: false, reason: 'revision-changed' })
    expect(service.applied).toEqual([])
  })

  it('a reconnect that redelivers the receipt gets the recorded outcome, not a second action', async () => {
    const service = new ReferenceService(hermesHome)
    const receipt = approved(await confirmInDesktop(service.propose(), 'Confirm'))

    const first = service.deliver(receipt)

    expect(service.deliver(receipt)).toEqual(first)
    expect(service.deliver(structuredClone(receipt))).toEqual(first)
    expect(service.applied).toHaveLength(1)
  })

  it('cancel yields nothing to deliver', async () => {
    const service = new ReferenceService(hermesHome)

    expect(await confirmInDesktop(service.propose(), 'Cancel')).toEqual({ status: 'cancelled' })
    expect(service.applied).toEqual([])
  })

  it('a receipt for a proposal the service never issued is rejected', async () => {
    const service = new ReferenceService(hermesHome)
    const other = new ReferenceService(hermesHome)
    const receipt = approved(await confirmInDesktop(other.propose(), 'Confirm'))

    expect(service.deliver(receipt)).toEqual({ ok: false, reason: 'not-issued' })
  })

  it('a receipt whose payload was rewritten is rejected', async () => {
    const service = new ReferenceService(hermesHome)
    const proposal = service.propose()
    const receipt = approved(await confirmInDesktop(proposal, 'Confirm'))
    service.edit()

    // A plugin rewriting the revision to match the new document breaks the signature.
    const forged = { ...receipt, payload: receipt.payload.replace('"revision":"r1"', '"revision":"r2"') }

    expect(service.deliver(forged)).toEqual({ ok: false, reason: 'bad-signature' })
    expect(service.applied).toEqual([])
  })
})
