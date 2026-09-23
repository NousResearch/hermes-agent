// "Send Diagnostics" — the error card's consent-gated debug-bundle upload.
//
// Flow: an error card (or any surface) calls requestSendDiagnostics() with
// optional error context → the modal host renders the privacy notice → the
// user explicitly clicks Upload → diagnostics.share_nous runs backend-side
// (collect + force-redact + Nous-S3 upload) → the modal shows the private
// view link plus the support handoff (GitHub Issues · Nous Portal Support ·
// Discord).
//
// Consent is per-upload and explicit — no "always allow", mirroring the CLI's
// `hermes debug share --nous` confirmation contract. On a remote connection
// the backend bundles ITS OWN logs (the runtime that owns the failure); the
// local desktop.log is attached as a client-side extra so support sees both
// halves in one bundle.
import { atom } from 'nanostores'

import { $gateway } from '@/store/gateway'

export interface SendDiagnosticsResult {
  expiresAt?: string
  uploadId?: string
  viewUrl?: string
}

export interface SendDiagnosticsState {
  /** Short text describing the failure that prompted the report (attached
   *  to the bundle as error-context.txt, redacted server-side). */
  errorContext?: string
  error?: string
  phase: 'consent' | 'done' | 'error' | 'uploading'
  result?: SendDiagnosticsResult
}

export const $sendDiagnostics = atom<SendDiagnosticsState | null>(null)

// Generation token: bumped on every open AND every dismiss. An in-flight
// upload captures the generation it started under and only writes its
// completion back when the token still matches — so dismissing mid-upload is
// immediate and a stale completion can't resurrect or overwrite the dialog.
// Request cancellation stays best-effort (the WS call runs to completion
// server-side; we just ignore the result).
let generation = 0

/** Open the consent modal. No network I/O happens until the user confirms. */
export function requestSendDiagnostics(errorContext?: string): void {
  generation += 1
  $sendDiagnostics.set({ errorContext, phase: 'consent' })
}

export function dismissSendDiagnostics(): void {
  generation += 1
  $sendDiagnostics.set(null)
}

interface ShareNousResponse {
  error?: string
  expires_at?: string
  ok: boolean
  upload_id?: string
  view_url?: string
}

/** Read the LOCAL desktop log via Electron so a remote backend's bundle still
 *  carries the Desktop-side transport evidence. Best-effort: absence of the
 *  IPC (browser dashboard, older shells) just omits the file. */
async function collectLocalExtras(): Promise<Record<string, string>> {
  try {
    const logs = await window.hermesDesktop?.getRecentLogs?.()
    const lines = Array.isArray(logs?.lines) ? logs.lines : []

    return lines.length ? { 'desktop.log': lines.join('\n') } : {}
  } catch {
    return {}
  }
}

// Bundle collection + upload legitimately takes a while (log reads + gzip +
// S3 leg); the default WS timeout is too tight for slow disks/links.
const SHARE_TIMEOUT_MS = 120_000

/** User confirmed — run the upload. Transitions consent → uploading → done/error. */
export async function confirmSendDiagnostics(): Promise<void> {
  const current = $sendDiagnostics.get()

  if (!current || current.phase !== 'consent') {
    return
  }

  const startedGeneration = generation

  // Only write back while the dialog the upload belongs to is still open.
  const stillCurrent = () => generation === startedGeneration

  $sendDiagnostics.set({ ...current, phase: 'uploading' })

  try {
    const gateway = $gateway.get()

    if (!gateway) {
      throw new Error('Hermes gateway unavailable')
    }

    const extraFiles = await collectLocalExtras()

    if (!stillCurrent()) {
      return
    }

    const response = await gateway.request<ShareNousResponse>(
      'diagnostics.share_nous',
      {
        ...(current.errorContext ? { error_context: current.errorContext } : {}),
        ...(Object.keys(extraFiles).length ? { extra_files: extraFiles } : {})
      },
      SHARE_TIMEOUT_MS
    )

    if (!stillCurrent()) {
      return
    }

    if (!response.ok) {
      throw new Error(response.error || 'upload failed')
    }

    $sendDiagnostics.set({
      ...current,
      phase: 'done',
      result: {
        expiresAt: response.expires_at,
        uploadId: response.upload_id,
        viewUrl: response.view_url
      }
    })
  } catch (error) {
    if (!stillCurrent()) {
      return
    }

    $sendDiagnostics.set({
      ...current,
      error: error instanceof Error ? error.message : String(error),
      phase: 'error'
    })
  }
}
