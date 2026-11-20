/**
 * `hermes:pluginConfirm:request`: only a Confirm click in main's own native
 * dialog yields a receipt, and the receipt verifies against the published key.
 */
import { createPublicKey, verify } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterAll, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  CONFIRMATION_CHANNEL,
  type ConfirmationResult,
  PUBLIC_KEY_FILE,
  registerPluginConfirmationIpc
} from './plugin-confirmation'

const hermesHome = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-plugin-confirm-')))
const handlers = new Map<string, (...args: unknown[]) => Promise<ConfirmationResult>>()
const showMessageBox = vi.fn()
const window = { id: 1 }
let windowFor: () => unknown = () => window

registerPluginConfirmationIpc({
  ipcMain: { handle: (channel: string, handler: never) => handlers.set(channel, handler) } as never,
  dialog: { showMessageBox } as never,
  windowFor: () => windowFor() as never,
  hermesHome
})

const proposal = {
  actionId: 'doc-42:replace-section',
  revision: 'r7',
  profile: 'default',
  serviceInstance: 'docsvc@3f9a',
  nonce: 'n0nce-AAAAAAAAAAAAAAAA',
  title: 'Replace section 2',
  detail: 'Rewrites "Pricing" in notes.md'
}

const call = (pluginId: unknown, value: unknown) => handlers.get(CONFIRMATION_CHANNEL)!({ sender: {} }, pluginId, value)

function verifyWithPublishedKey(receipt: { payload: string; signature: string; keyId: string }) {
  const key = JSON.parse(fs.readFileSync(path.join(hermesHome, PUBLIC_KEY_FILE), 'utf8'))

  return (
    key.keyId === receipt.keyId &&
    verify(
      null,
      Buffer.from(receipt.payload, 'utf8'),
      createPublicKey(key.publicKey),
      Buffer.from(receipt.signature, 'base64')
    )
  )
}

afterAll(() => fs.rmSync(hermesHome, { recursive: true, force: true }))

beforeEach(() => {
  showMessageBox.mockReset()
  windowFor = () => window
})

describe('plugin action confirmation', () => {
  it('Confirm in the native dialog yields a receipt the published key verifies', async () => {
    showMessageBox.mockResolvedValueOnce({ response: 1, checkboxChecked: false })

    const result = await call('doc-sync', proposal)

    expect(result.status).toBe('approved')
    const { receipt } = result as Extract<ConfirmationResult, { status: 'approved' }>
    expect(verifyWithPublishedKey(receipt)).toBe(true)
    expect(JSON.parse(receipt.payload)).toMatchObject({
      type: 'hermes.desktop.action-confirmation',
      decision: 'approved',
      plugin: 'doc-sync',
      ...proposal
    })

    // The dialog is attached to the asking window and shows every bound id.
    const [parent, options] = showMessageBox.mock.calls[0]
    expect(parent).toBe(window)
    expect(options).toMatchObject({ message: proposal.title, defaultId: 0, cancelId: 0 })

    for (const shown of [proposal.actionId, proposal.revision, proposal.profile, proposal.serviceInstance]) {
      expect(options.detail).toContain(shown)
    }
  })

  it('a tampered payload no longer verifies', async () => {
    showMessageBox.mockResolvedValueOnce({ response: 1, checkboxChecked: false })

    const { receipt } = (await call('doc-sync', proposal)) as Extract<ConfirmationResult, { status: 'approved' }>

    expect(verifyWithPublishedKey({ ...receipt, payload: receipt.payload.replace('"r7"', '"r8"') })).toBe(false)
  })

  it('Cancel yields no receipt', async () => {
    showMessageBox.mockResolvedValueOnce({ response: 0, checkboxChecked: false })

    expect(await call('doc-sync', proposal)).toEqual({ status: 'cancelled' })
  })

  it('the dialog closes itself at expiry and yields no receipt', async () => {
    vi.useFakeTimers()

    try {
      showMessageBox.mockImplementationOnce(
        (_parent, options: { signal: AbortSignal }) =>
          new Promise(resolve => options.signal.addEventListener('abort', () => resolve({ response: 0 })))
      )

      const pending = call('doc-sync', { ...proposal, expiresInMs: 5_000 })
      await vi.advanceTimersByTimeAsync(5_000)

      expect(await pending).toEqual({ status: 'expired' })
    } finally {
      vi.useRealTimers()
    }
  })

  it('a second request while a dialog is open is busy, not queued', async () => {
    let answer: (value: { response: number }) => void = () => undefined
    showMessageBox.mockImplementationOnce(() => new Promise(resolve => (answer = resolve)))

    const first = call('doc-sync', proposal)

    expect(await call('doc-sync', { ...proposal, nonce: 'other-nonce-BBBBBBBBBB' })).toEqual({ status: 'busy' })
    answer({ response: 0 })
    expect(await first).toEqual({ status: 'cancelled' })
    expect(showMessageBox).toHaveBeenCalledTimes(1)
  })

  it.each([
    ['a missing nonce', { ...proposal, nonce: undefined }],
    ['a short nonce', { ...proposal, nonce: 'short' }],
    ['an id with whitespace', { ...proposal, revision: 'r7 ' }],
    ['an invalid profile', { ...proposal, profile: '../etc' }],
    ['an empty title', { ...proposal, title: '   ' }],
    ['a non-object', 'approve everything']
  ])('%s is refused without opening a dialog', async (_label, value) => {
    expect((await call('doc-sync', value)).status).toBe('refused')
    expect(showMessageBox).not.toHaveBeenCalled()
  })

  it('no app window means no dialog', async () => {
    windowFor = () => null

    expect((await call('doc-sync', proposal)).status).toBe('refused')
    expect(showMessageBox).not.toHaveBeenCalled()
  })

  it('control and bidi characters cannot hide text from the dialog', async () => {
    showMessageBox.mockResolvedValueOnce({ response: 1, checkboxChecked: false })

    const result = (await call('doc-sync', {
      ...proposal,
      title: 'Pay \u202Eevil\u202C\nnow'
    })) as Extract<ConfirmationResult, { status: 'approved' }>

    const shown = showMessageBox.mock.calls[0][1].message
    expect(shown).toBe('Pay  evil  now')
    expect(JSON.parse(result.receipt.payload).title).toBe(shown)
  })
})
