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
//
// The upload is routed like any other profile-scoped RPC: to the profile the
// dialog was opened for (the failing session's owner, else the focused
// profile), with that session's runtime id. A multiplexed backend serves
// several profiles on one socket, and an unscoped call there would bundle the
// LAUNCH profile's logs instead of the ones the user agreed to share.
import { atom } from 'nanostores'

import { activeGatewayProfileKey, requestGatewayForAgent } from '@/store/gateway'

export interface SendDiagnosticsResult {
  expiresAt?: string
  uploadId?: string
  viewUrl?: string
}

/** Who owns the failure: the session's owner route and runtime id. */
export interface SendDiagnosticsOwner {
  connectionId?: null | string
  profile?: null | string
  sessionId?: null | string
}

export interface SendDiagnosticsState {
  /** Registry connection of the owning session (cross-connection Bot chats). */
  connectionId?: string
  /** Short text describing the failure that prompted the report (attached
   *  to the bundle as error-context.txt, redacted server-side). */
  errorContext?: string
  error?: string
  phase: 'consent' | 'done' | 'error' | 'uploading'
  /** Profile whose logs are bundled, fixed when the consent dialog opens. */
  profile: string
  result?: SendDiagnosticsResult
  /** Runtime session the failure came from; the backend scopes to its profile. */
  sessionId?: string
}

export const $sendDiagnostics = atom<SendDiagnosticsState | null>(null)

// Generation token: bumped on every open AND every dismiss. An in-flight
// upload captures the generation it started under and only writes its
// completion back when the token still matches — so dismissing mid-upload is
// immediate and a stale completion can't resurrect or overwrite the dialog.
// Request cancellation stays best-effort (the WS call runs to completion
// server-side; we just ignore the result).
let generation = 0

/** Open the consent modal. No network I/O happens until the user confirms.
 *  The target profile is fixed here, so switching profiles while the dialog is
 *  open cannot redirect an upload the user consented to for another one. */
export function requestSendDiagnostics(errorContext?: string, owner: SendDiagnosticsOwner = {}): void {
  const connectionId = owner.connectionId?.trim()
  const sessionId = owner.sessionId?.trim()

  generation += 1
  $sendDiagnostics.set({
    errorContext,
    phase: 'consent',
    profile: owner.profile?.trim() || activeGatewayProfileKey(),
    ...(connectionId ? { connectionId } : {}),
    ...(sessionId ? { sessionId } : {})
  })
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
    const extraFiles = await collectLocalExtras()

    if (!stillCurrent()) {
      return
    }

    // requestGatewayForAgent picks the socket that serves this (connection,
    // profile) and adds `profile` when that socket is a shared multi-profile
    // backend; `session_id` lets the backend scope to the session's owner.
    const response = await requestGatewayForAgent<ShareNousResponse>(
      current.connectionId ?? null,
      current.profile,
      'diagnostics.share_nous',
      {
        ...(current.errorContext ? { error_context: current.errorContext } : {}),
        ...(Object.keys(extraFiles).length ? { extra_files: extraFiles } : {}),
        ...(current.sessionId ? { session_id: current.sessionId } : {})
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
