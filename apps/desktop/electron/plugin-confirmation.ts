// The plugin SDK's confirmation door (`ctx.confirmAction`, #158148). A local
// service proposes one action; the user confirms or cancels it in a NATIVE
// dialog that main opens itself, and only a confirmation yields a receipt main
// signs with a per-launch Ed25519 key. The renderer, where plugin code runs,
// can open the dialog but can neither click it nor sign, so a verified receipt
// is evidence the user confirmed exactly the action, revision, profile and
// service instance the dialog showed. The service verifies the signature
// against the public key file main publishes under HERMES_HOME, then owns
// everything after: single use per nonce, the receipt record, idempotent
// replies on redelivery.

import { createHash, generateKeyPairSync, type KeyObject, sign } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import type { BrowserWindow, Dialog, IpcMain, IpcMainInvokeEvent, MessageBoxOptions } from 'electron'

import { DESKTOP_PROFILE_NAME_RE } from './desktop-profile'

export const CONFIRMATION_CHANNEL = 'hermes:pluginConfirm:request'
export const RECEIPT_TYPE = 'hermes.desktop.action-confirmation'
export const RECEIPT_VERSION = 1
/** Relative to HERMES_HOME; the one place a service looks for the key. */
export const PUBLIC_KEY_FILE = path.join('desktop', 'action-confirmation-key.json')

const DEFAULT_EXPIRY_MS = 120_000
const MIN_EXPIRY_MS = 5_000
const MAX_EXPIRY_MS = 600_000
const TITLE_MAX = 200
const DETAIL_MAX = 2_000
// Ids are shown verbatim in the dialog and signed verbatim, so they stay
// printable and bounded: no control characters, no look-alike whitespace.
const TOKEN_RE = /^[A-Za-z0-9._:@/+=-]{1,200}$/
const NONCE_RE = /^[A-Za-z0-9_-]{16,128}$/
const PLUGIN_ID_RE = /^[a-z0-9][a-z0-9._-]{0,63}$/i

export interface ActionProposal {
  actionId: string
  revision: string
  profile: string
  serviceInstance: string
  nonce: string
  title: string
  detail: string
  expiresInMs: number
}

export interface ConfirmationReceipt {
  v: typeof RECEIPT_VERSION
  type: typeof RECEIPT_TYPE
  keyId: string
  decision: 'approved'
  plugin: string
  actionId: string
  revision: string
  profile: string
  serviceInstance: string
  nonce: string
  title: string
  detail: string
  confirmedAt: string
  expiresAt: string
}

/** `payload` is the exact signed UTF-8 JSON; verify it before parsing it. */
export interface SignedReceipt {
  payload: string
  signature: string
  keyId: string
}

export type ConfirmationResult =
  | { status: 'approved'; receipt: SignedReceipt }
  | { status: 'cancelled' | 'expired' | 'busy' }
  | { status: 'refused'; error: string }

export interface ConfirmationSigner {
  keyId: string
  publicKeyFile: string
  sign: (payload: string) => string
}

export interface PluginConfirmationDeps {
  ipcMain: Pick<IpcMain, 'handle'>
  dialog: Pick<Dialog, 'showMessageBox'>
  windowFor: (event: IpcMainInvokeEvent) => BrowserWindow | null
  hermesHome: string
  now?: () => number
}

// Control, line-separator, bidi-mark and bidi-override characters would let
// the dialog show something other than what is signed; they become spaces.
// Only the detail keeps its line breaks.
// eslint-disable-next-line no-control-regex -- matching control characters is the point
const UNSAFE_INLINE = /[\u0000-\u001F\u007F-\u009F\u061C\u200E\u200F\u2028\u2029\u202A-\u202E\u2066-\u2069]/g

const UNSAFE_MULTILINE =
  // eslint-disable-next-line no-control-regex -- matching control characters is the point
  /[\u0000-\u0009\u000B-\u001F\u007F-\u009F\u061C\u200E\u200F\u2028\u2029\u202A-\u202E\u2066-\u2069]/g

const text = (value: unknown, max: number, unsafe: RegExp) =>
  typeof value === 'string' ? value.replace(unsafe, ' ').trim().slice(0, max) : ''

function token(raw: Record<string, unknown>, key: string, pattern: RegExp): string {
  const value = raw[key]

  if (typeof value !== 'string' || !pattern.test(value)) {
    throw new Error(`Invalid ${key}.`)
  }

  return value
}

/** Parse a renderer-supplied proposal; throws with the first bad field. */
export function parseProposal(value: unknown): ActionProposal {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('A proposal object is required.')
  }

  const raw = value as Record<string, unknown>
  const title = text(raw.title, TITLE_MAX, UNSAFE_INLINE)

  if (!title) {
    throw new Error('Invalid title.')
  }

  if (typeof raw.profile !== 'string' || !DESKTOP_PROFILE_NAME_RE.test(raw.profile)) {
    throw new Error('Invalid profile.')
  }

  const expiry = raw.expiresInMs === undefined ? DEFAULT_EXPIRY_MS : raw.expiresInMs

  if (typeof expiry !== 'number' || !Number.isFinite(expiry)) {
    throw new Error('Invalid expiresInMs.')
  }

  return {
    actionId: token(raw, 'actionId', TOKEN_RE),
    revision: token(raw, 'revision', TOKEN_RE),
    profile: raw.profile,
    serviceInstance: token(raw, 'serviceInstance', TOKEN_RE),
    nonce: token(raw, 'nonce', NONCE_RE),
    title,
    detail: text(raw.detail, DETAIL_MAX, UNSAFE_MULTILINE),
    expiresInMs: Math.min(MAX_EXPIRY_MS, Math.max(MIN_EXPIRY_MS, Math.round(expiry)))
  }
}

/** A fresh per-launch key whose public half is published under HERMES_HOME.
 *  The private key never leaves this process and never touches disk. */
export function createConfirmationSigner(hermesHome: string, now: () => number = Date.now): ConfirmationSigner {
  const { privateKey, publicKey }: { privateKey: KeyObject; publicKey: KeyObject } = generateKeyPairSync('ed25519')
  const spki = publicKey.export({ format: 'der', type: 'spki' })
  const keyId = createHash('sha256').update(spki).digest('hex').slice(0, 32)
  const publicKeyFile = path.join(hermesHome, PUBLIC_KEY_FILE)

  const record = {
    v: RECEIPT_VERSION,
    alg: 'Ed25519',
    keyId,
    publicKey: publicKey.export({ format: 'pem', type: 'spki' }).toString(),
    createdAt: new Date(now()).toISOString(),
    pid: process.pid
  }

  // Atomic replace, so a service never reads a half-written key.
  fs.mkdirSync(path.dirname(publicKeyFile), { recursive: true })
  const staging = `${publicKeyFile}.${process.pid}.tmp`
  fs.writeFileSync(staging, `${JSON.stringify(record, null, 2)}\n`, { mode: 0o644 })
  fs.renameSync(staging, publicKeyFile)

  return {
    keyId,
    publicKeyFile,
    sign: payload => sign(null, Buffer.from(payload, 'utf8'), privateKey).toString('base64')
  }
}

export function confirmationDialogOptions(pluginId: string, proposal: ActionProposal): MessageBoxOptions {
  const facts = [
    `Action: ${proposal.actionId}`,
    `Revision: ${proposal.revision}`,
    `Profile: ${proposal.profile}`,
    `Service: ${proposal.serviceInstance}`
  ].join('\n')

  return {
    type: 'question',
    title: 'Confirm action',
    message: proposal.title,
    detail: `${proposal.detail ? `${proposal.detail}\n\n` : ''}${facts}\n\nRequested by plugin "${pluginId}".`,
    buttons: ['Cancel', 'Confirm'],
    // Cancel is the default and the Escape answer: a stray Enter never confirms.
    defaultId: 0,
    cancelId: 0,
    noLink: true
  }
}

/** One confirmation at a time: a dialog already open answers `busy`. */
export function registerPluginConfirmationIpc(deps: PluginConfirmationDeps): void {
  const now = deps.now ?? Date.now
  // Published now, not at the first approval: until this launch's key replaces
  // the file, the previous launch's key would still verify its receipts. A key
  // that cannot be published refuses every request, so no receipt is signed by
  // a key no service can read.
  let signer: ConfirmationSigner | null = null
  let signerError = ''

  try {
    signer = createConfirmationSigner(deps.hermesHome, now)
  } catch (error) {
    signerError = `Could not publish the confirmation key: ${error instanceof Error ? error.message : String(error)}`
  }

  let open = false

  deps.ipcMain.handle(
    CONFIRMATION_CHANNEL,
    async (event: IpcMainInvokeEvent, pluginId: unknown, value: unknown): Promise<ConfirmationResult> => {
      let proposal: ActionProposal

      try {
        if (typeof pluginId !== 'string' || !PLUGIN_ID_RE.test(pluginId)) {
          throw new Error('Invalid plugin id.')
        }

        proposal = parseProposal(value)
      } catch (error) {
        return { status: 'refused', error: error instanceof Error ? error.message : String(error) }
      }

      if (!signer) {
        return { status: 'refused', error: signerError }
      }

      const window = deps.windowFor(event)

      // Without a parent the dialog could open behind another app, and the
      // expiry abort does not reach a parentless macOS message box.
      if (!window) {
        return { status: 'refused', error: 'No app window to show the confirmation in.' }
      }

      if (open) {
        return { status: 'busy' }
      }

      open = true
      const expiresAt = now() + proposal.expiresInMs
      const abort = new AbortController()
      const timer = setTimeout(() => abort.abort(), proposal.expiresInMs)

      try {
        const { response } = await deps.dialog.showMessageBox(window, {
          ...confirmationDialogOptions(pluginId, proposal),
          signal: abort.signal
        })

        const confirmedAt = now()

        if (abort.signal.aborted || confirmedAt >= expiresAt) {
          return { status: 'expired' }
        }

        if (response !== 1) {
          return { status: 'cancelled' }
        }

        const receipt: ConfirmationReceipt = {
          v: RECEIPT_VERSION,
          type: RECEIPT_TYPE,
          keyId: signer.keyId,
          decision: 'approved',
          plugin: pluginId,
          actionId: proposal.actionId,
          revision: proposal.revision,
          profile: proposal.profile,
          serviceInstance: proposal.serviceInstance,
          nonce: proposal.nonce,
          title: proposal.title,
          detail: proposal.detail,
          confirmedAt: new Date(confirmedAt).toISOString(),
          expiresAt: new Date(expiresAt).toISOString()
        }

        const payload = JSON.stringify(receipt)

        return { status: 'approved', receipt: { payload, signature: signer.sign(payload), keyId: signer.keyId } }
      } catch (error) {
        return { status: 'refused', error: error instanceof Error ? error.message : String(error) }
      } finally {
        clearTimeout(timer)
        open = false
      }
    }
  )
}
