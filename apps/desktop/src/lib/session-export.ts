import type { SessionInfo } from '@/hermes'
import { getAllSessionMessages } from '@/hermes'
import { translateNow } from '@/i18n'
import { runtimeTranslations } from '@/i18n/runtime'
import { type SaveFormat, saveTranscript } from '@/lib/save-transcript'
import type { TranscriptOptions } from '@/lib/transcript-document'
import { activeGateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { ambientRequestFor } from '@/store/session-gone-latch'
import { runtimeIdForStoredSession } from '@/store/session-runtime-binding'
import { requestForOwnedSession } from '@/store/session-states'

interface ExportSessionParams {
  sessionId: string
  profile?: string | null
  title?: string | null
  session?: SessionInfo
}

export interface ExportSessionFileTarget extends ExportSessionParams {
  format: SaveFormat
  options?: TranscriptOptions
  redact: boolean
}

export type ExportSessionOutcome =
  | { kind: 'saved'; path: string }
  | { kind: 'canceled' }
  | { kind: 'json-download' }
  | { kind: 'needs-live-backend' }
  | { kind: 'unsupported' }
  | { kind: 'failed'; error: unknown }

export async function exportSessionFile(target: ExportSessionFileTarget): Promise<ExportSessionOutcome> {
  if (!target.sessionId) {
    return { kind: 'failed', error: new Error('not_found') }
  }

  const runtimeId = runtimeIdForStoredSession(target.sessionId)
  const gateway = activeGateway()

  const fallback = async (): Promise<ExportSessionOutcome> => {
    // A legacy JSON download must never silently discard a redaction request.
    if (target.format !== 'json' || target.redact || target.options) {
      return { kind: 'needs-live-backend' }
    }

    try {
      await downloadSessionJson(target.sessionId, target)

      return { kind: 'json-download' }
    } catch (error) {
      return { kind: 'failed', error }
    }
  }

  if (!runtimeId || !gateway) {
    return fallback()
  }

  const outcome = await saveTranscript({
    format: target.format,
    options: target.options,
    redact: target.redact,
    requestGateway: (method, params) => requestForOwnedSession(runtimeId, ambientRequestFor(gateway), method, params),
    sessionId: runtimeId,
    title: runtimeTranslations().promptNotices.desktopCommands.saveDialogTitle
  })

  if (outcome.kind === 'missing-export') {
    return fallback()
  }

  return outcome.kind === 'error' ? { kind: 'failed', error: outcome.error } : outcome
}

function sanitizeFilenamePart(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)
}

function sessionExportFilename(sessionId: string, title?: string | null) {
  const titlePart = title ? sanitizeFilenamePart(title) : ''
  const idPart = sanitizeFilenamePart(sessionId).slice(0, 8) || 'session'

  return `${titlePart || 'session'}-${idPart}.json`
}

async function downloadSessionJson(sessionId: string, params: Omit<ExportSessionParams, 'sessionId'> = {}) {
  const profile = params.profile ?? params.session?.profile
  const { messages } = await getAllSessionMessages(sessionId, profile)

  const payload = {
    exported_at: new Date().toISOString(),
    session_id: sessionId,
    title: params.title ?? null,
    session: params.session ?? null,
    message_count: messages.length,
    messages
  }

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const downloadUrl = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = downloadUrl
  anchor.download = sessionExportFilename(sessionId, params.title)
  anchor.click()
  URL.revokeObjectURL(downloadUrl)
}

export async function exportSession(sessionId: string, params: Omit<ExportSessionParams, 'sessionId'> = {}) {
  if (!sessionId) {
    return
  }

  try {
    await downloadSessionJson(sessionId, params)
    notify({ kind: 'success', message: translateNow('desktop.sessionExported'), durationMs: 2_000 })
  } catch (err) {
    notifyError(err, translateNow('desktop.sessionExportFailed'))
  }
}
