import type { SessionInfo } from '@/hermes'
import { getAllSessionMessages } from '@/hermes'
import { translateNow } from '@/i18n'
import { writeClipboardText } from '@/lib/clipboard'
import { notify, notifyError } from '@/store/notifications'

interface ExportSessionParams {
  sessionId: string
  profile?: string | null
  title?: string | null
  session?: SessionInfo
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

    notify({ kind: 'success', message: translateNow('desktop.sessionExported'), durationMs: 2_000 })
  } catch (err) {
    notifyError(err, translateNow('desktop.sessionExportFailed'))
  }
}

type ExportRequest = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

/** "Copy concise transcript": the OWNER renders the whole conversation's visible dialogue as Markdown from its
 *  stored display history (`session.export` + `concise`: compaction-archived turns and the compression lineage
 *  included; no tool logs, no model call), and THIS client's clipboard receives it — never the gateway host's,
 *  which may be another machine. An owner that predates the option refuses it; that is reported, never answered
 *  from the rows loaded in the UI. Success is shown only after the clipboard write resolved. */
export async function copyConciseTranscript(runtimeSessionId: null | string | undefined, request: ExportRequest) {
  if (!runtimeSessionId) {
    notify({ kind: 'error', message: translateNow('desktop.conciseTranscriptNeedsOpenSession') })

    return
  }

  try {
    const result = await request<{ content?: string }>('session.export', {
      concise: true,
      format: 'md',
      session_id: runtimeSessionId
    })

    const content = result?.content

    if (typeof content !== 'string' || !content.trim()) {
      notify({ kind: 'error', message: translateNow('desktop.conciseTranscriptEmpty') })

      return
    }

    await writeClipboardText(content)
    notify({ kind: 'success', message: translateNow('desktop.conciseTranscriptCopied'), durationMs: 2_000 })
  } catch (err) {
    const outdated = err instanceof Error && (err.message === 'invalid_params' || err.message === 'unknown_method')

    outdated
      ? notify({ kind: 'error', message: translateNow('desktop.conciseTranscriptOutdated') })
      : notifyError(err, translateNow('desktop.conciseTranscriptFailed'))
  }
}
