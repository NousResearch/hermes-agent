import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { parseTranscriptDocument, type TranscriptOptions } from '@/lib/transcript-document'
import { renderTranscript } from '@/lib/transcript-render'

export type SaveFormat = 'docx' | 'odt' | 'html' | 'json' | 'md'

const SAVE_FILTERS: Record<SaveFormat, { extensions: string[]; name: string }> = {
  docx: { extensions: ['docx'], name: 'Word document' },
  odt: { extensions: ['odt'], name: 'OpenDocument text' },
  html: { extensions: ['html'], name: 'HTML' },
  json: { extensions: ['json'], name: 'JSON' },
  md: { extensions: ['md'], name: 'Markdown' }
}

export type SaveTranscriptOutcome =
  | { kind: 'saved'; path: string }
  | { kind: 'canceled' }
  | { kind: 'missing-export' }
  | { kind: 'unsupported' }
  | { kind: 'error'; error: unknown }

interface SaveTranscriptOptions {
  filename?: string
  options?: TranscriptOptions
  format: SaveFormat
  redact: boolean
  requestGateway: (method: string, params?: Record<string, unknown>) => Promise<unknown>
  sessionId: string
  title: string
}

interface RenderedExport {
  content?: unknown
  filename?: unknown
  format?: unknown
  transcript_version?: unknown
}

function exportFilename(suggested: string | undefined, format: SaveFormat, filtered: boolean): string | undefined {
  if (!filtered) {return suggested}

  return `${(suggested || 'conversation').replace(/\.[^.\\/]+$/, '')}.${format}`
}

function prepareExport(
  content: string,
  format: SaveFormat,
  options?: TranscriptOptions
): null | { content: string; encoding?: 'base64' } {
  const document = options ? parseTranscriptDocument(content) : undefined

  if (
    document &&
    options &&
    Object.entries(options).some(([key, value]) => document.options[key as keyof TranscriptOptions] !== value)
  )
    {return null}

  const rendered = document ? renderTranscript(document, format) : { content }

  if (
    'encoding' in rendered &&
    rendered.encoding === 'base64' &&
    !window.hermesDesktop?.saveTextFileEncodings?.includes('base64')
  )
    {return null}

  return rendered
}

// The owner projects the transcript; the client chooses and writes its own file.
// A remote gateway must never choose a path on this machine.
export async function saveTranscript({
  filename,
  format,
  options,
  redact,
  requestGateway,
  sessionId,
  title
}: SaveTranscriptOptions): Promise<SaveTranscriptOutcome> {
  let file: null | RenderedExport
  const wireFormat = options ? 'json' : format

  try {
    file = (await requestGateway('session.export', {
      session_id: sessionId,
      format: wireFormat,
      ...(options ? { transcript_options: options } : {}),
      ...(redact ? { redact: true } : {})
    })) as null | RenderedExport
  } catch (error) {
    return isMissingRpcMethod(error) ? { kind: 'missing-export' } : { kind: 'error', error }
  }

  const save = window.hermesDesktop?.saveTextFileAs

  if (
    typeof file?.content !== 'string' ||
    typeof save !== 'function' ||
    (options && file.transcript_version !== 1) ||
    (file.format !== undefined && file.format !== wireFormat)
  ) {
    return { kind: 'unsupported' }
  }

  try {
    const rendered = prepareExport(file.content, format, options)

    if (!rendered) {return { kind: 'unsupported' }}
    const suggested = filename || (typeof file.filename === 'string' ? file.filename : undefined)

    const result = await save({
      ...rendered,
      defaultPath: exportFilename(suggested, format, options !== undefined),
      filters: [SAVE_FILTERS[format]],
      title
    })

    if (result?.canceled) {
      return { kind: 'canceled' }
    }

    return result?.saved && result.path
      ? { kind: 'saved', path: result.path }
      : { kind: 'error', error: new Error('export_save_failed') }
  } catch (error) {
    return { kind: 'error', error }
  }
}
