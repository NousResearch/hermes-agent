import type { SessionInfo } from '@/hermes'
import { getSessionMessages } from '@/hermes'
import { translateNow } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'

interface ExportSessionParams {
  sessionId: string
  title?: string | null
  session?: SessionInfo
  profile?: string | null
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

export async function exportSession(sessionId: string, params: Omit<ExportSessionParams, 'sessionId'> = {}) {
  if (!sessionId) {
    return
  }

  try {
    // Route the transcript read to the session's owning profile. A remote (or
    // non-default local) profile's rows live only on that profile's backend, so
    // reading without it falls through to the local primary -> 404/empty/wrong
    // transcript. Matches the resume/list/rename read-routing from #39894.
    const profile = params.profile ?? params.session?.profile ?? null
    const { messages } = await getSessionMessages(sessionId, profile)

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

    notify({ kind: 'success', message: translateNow('desktop.sessionExported'), durationMs: 2_000 })
  } catch (err) {
    notifyError(err, translateNow('desktop.sessionExportFailed'))
  }
}
