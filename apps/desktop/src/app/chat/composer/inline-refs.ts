import { formatRefValue } from '@/components/assistant-ui/directive-text'
import { contextPath } from '@/lib/chat-runtime'

import type { DroppedFile } from '../hooks/use-composer-actions'

import { composerPlainText, escapeHtml, placeCaretEnd, refChipHtml } from './rich-editor'

/** A chip to insert: a raw `@kind:value` string, or a typed value + display label. */
export type InlineRefInput = string | { kind: string; label?: string; value: string }

/** MIME for an in-app session drag (sidebar row → composer). */
export const HERMES_SESSION_MIME = 'application/x-hermes-session'

export interface SessionDragPayload {
  id: string
  profile: string
  title: string
}

export function writeSessionDrag(transfer: DataTransfer, payload: SessionDragPayload) {
  transfer.setData(HERMES_SESSION_MIME, JSON.stringify(payload))
  transfer.effectAllowed = 'copy'
}

export function dragHasSession(transfer: DataTransfer | null) {
  return Boolean(transfer) && Array.from(transfer!.types || []).includes(HERMES_SESSION_MIME)
}

export function readSessionDrag(transfer: DataTransfer | null): null | SessionDragPayload {
  const raw = transfer?.getData(HERMES_SESSION_MIME)

  if (!raw) {
    return null
  }

  try {
    const parsed = JSON.parse(raw) as Partial<SessionDragPayload>

    return parsed.id ? { id: parsed.id, profile: parsed.profile || 'default', title: parsed.title || '' } : null
  } catch {
    return null
  }
}

/** Build a `@session:<profile>/<id>` chip. Value carries the metadata the agent
 * needs to resolve the link (session_search); label shows the friendly title. */
export function sessionInlineRef({ id, profile, title }: SessionDragPayload): InlineRefInput {
  return { kind: 'session', label: title || `chat ${id.slice(0, 8)}`, value: `${profile || 'default'}/${id}` }
}

export function dragHasAttachments(transfer: DataTransfer | null, pathsMime: string) {
  if (!transfer) {
    return false
  }

  if (Array.from(transfer.types || []).includes(pathsMime)) {
    return true
  }

  if (Array.from(transfer.types || []).includes('Files')) {
    return true
  }

  return Array.from(transfer.items || []).some(item => item.kind === 'file')
}

export function droppedFileInlineRef(candidate: DroppedFile, cwd: string | null | undefined) {
  if (!candidate.path) {
    return null
  }

  const rel = contextPath(candidate.path, cwd || '')

  if (candidate.line) {
    const { line, lineEnd } = candidate
    const range = lineEnd && lineEnd > line ? `${line}-${lineEnd}` : `${line}`

    return `@line:${formatRefValue(`${rel}:${range}`)}`
  }

  const kind = candidate.isDirectory ? 'folder' : 'file'

  return `@${kind}:${formatRefValue(rel)}`
}

export function insertInlineRefsIntoEditor(editor: HTMLDivElement, refs: readonly InlineRefInput[]) {
  if (!refs.length) {
    return null
  }

  const refsHtml = refs
    .map(ref => {
      if (typeof ref !== 'string') {
        return refChipHtml(ref.kind, ref.value, ref.label)
      }

      const match = ref.match(/^@([^:]+):(.+)$/)

      return match ? refChipHtml(match[1], match[2]) : escapeHtml(ref)
    })
    .join(' ')

  const selection = window.getSelection()

  const range =
    selection?.rangeCount && editor.contains(selection.getRangeAt(0).commonAncestorContainer)
      ? selection.getRangeAt(0)
      : null

  editor.focus({ preventScroll: true })

  if (range) {
    const beforeRange = range.cloneRange()
    beforeRange.selectNodeContents(editor)
    beforeRange.setEnd(range.startContainer, range.startOffset)
    const beforeContainer = document.createElement('div')
    beforeContainer.appendChild(beforeRange.cloneContents())

    const afterRange = range.cloneRange()
    afterRange.selectNodeContents(editor)
    afterRange.setStart(range.endContainer, range.endOffset)
    const afterContainer = document.createElement('div')
    afterContainer.appendChild(afterRange.cloneContents())

    const beforeText = composerPlainText(beforeContainer)
    const afterText = composerPlainText(afterContainer)
    const needsBeforeSpace = beforeText.length > 0 && !/\s$/.test(beforeText)
    const needsAfterSpace = afterText.length === 0 || !/^\s/.test(afterText)

    document.execCommand('insertHTML', false, `${needsBeforeSpace ? ' ' : ''}${refsHtml}${needsAfterSpace ? ' ' : ''}`)
  } else {
    const current = composerPlainText(editor)
    placeCaretEnd(editor)
    document.execCommand('insertHTML', false, `${current && !/\s$/.test(current) ? ' ' : ''}${refsHtml} `)
  }

  return composerPlainText(editor)
}
