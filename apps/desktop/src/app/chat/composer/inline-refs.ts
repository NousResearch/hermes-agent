import { formatRefValue } from '@/components/assistant-ui/directive-text'
import { translateNow } from '@/i18n'
import { contextPath } from '@/lib/chat-runtime'

import type { DroppedFile } from '../hooks/use-composer-actions'

import { composerPlainText, normalizeComposerEditorDom, placeCaretEnd, refChipElement } from './rich-editor'

/** A chip to insert: a raw `@kind:value` string, or a typed value + display label. */
export type InlineRefInput = string | { kind: string; label?: string; value: string }

/** A dragged sidebar session — carried in-memory by the pointer drag session
 *  (session-drag.ts); sessions never ride native DnD. */
export interface SessionDragPayload {
  id: string
  profile: string
  title: string
}

/** A session's friendly display label — its title, or a localized fallback. */
export const sessionLabel = ({ id, title }: SessionDragPayload) =>
  title || translateNow('sidebar.row.untitledChat', id.slice(0, 8))

/** Build a `@session:<profile>/<id>` chip. Value carries the metadata the agent
 * needs to resolve the link (session_search); label shows the friendly title. */
export function sessionInlineRef(payload: SessionDragPayload): InlineRefInput {
  return { kind: 'session', label: sessionLabel(payload), value: `${payload.profile || 'default'}/${payload.id}` }
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

/** Resolve a batch of drops to their inline `@file:`/`@line:`/`@folder:` refs,
 * dropping any that carry no path. */
export function droppedFileInlineRefs(candidates: DroppedFile[], cwd: string | null | undefined): string[] {
  return candidates.map(candidate => droppedFileInlineRef(candidate, cwd)).filter((ref): ref is string => Boolean(ref))
}

function parseInlineRef(ref: InlineRefInput): { kind: string; label?: string; rawValue: string } | null {
  if (typeof ref !== 'string') {
    return { kind: ref.kind, label: ref.label, rawValue: ref.value }
  }

  const match = ref.match(/^@([^:]+):(.+)$/)

  if (!match) {
    return null
  }

  return { kind: match[1] || 'file', rawValue: match[2] || '' }
}

function plainTextInRange(editor: HTMLDivElement, range: Range, edge: 'after' | 'before') {
  const slice = range.cloneRange()
  slice.selectNodeContents(editor)

  if (edge === 'before') {
    slice.setEnd(range.startContainer, range.startOffset)
  } else {
    slice.setStart(range.endContainer, range.endOffset)
  }

  const container = document.createElement('div')
  container.appendChild(slice.cloneContents())

  return composerPlainText(container)
}

function buildRefFragment(
  refs: readonly { kind: string; label?: string; rawValue: string }[],
  { needsBeforeSpace, needsAfterSpace }: { needsAfterSpace: boolean; needsBeforeSpace: boolean }
) {
  const fragment = document.createDocumentFragment()

  if (needsBeforeSpace) {
    fragment.append(document.createTextNode(' '))
  }

  refs.forEach((ref, index) => {
    if (index > 0) {
      fragment.append(document.createTextNode(' '))
    }

    fragment.append(refChipElement(ref.kind, ref.rawValue, ref.label))
  })

  if (needsAfterSpace) {
    fragment.append(document.createTextNode(' '))
  }

  return fragment
}

export function insertInlineRefsIntoEditor(editor: HTMLDivElement, refs: readonly InlineRefInput[]) {
  const parsed = refs.map(parseInlineRef).filter((ref): ref is NonNullable<typeof ref> => ref !== null)

  if (!parsed.length) {
    return null
  }

  editor.focus({ preventScroll: true })

  const selection = window.getSelection()

  const range =
    selection?.rangeCount && editor.contains(selection.getRangeAt(0).commonAncestorContainer)
      ? selection.getRangeAt(0)
      : null

  if (range && selection) {
    const beforeText = plainTextInRange(editor, range, 'before')
    const afterText = plainTextInRange(editor, range, 'after')

    range.insertNode(
      buildRefFragment(parsed, {
        needsAfterSpace: afterText.length === 0 || !/^\s/.test(afterText),
        needsBeforeSpace: beforeText.length > 0 && !/\s$/.test(beforeText)
      })
    )
    range.collapse(false)
    selection.removeAllRanges()
    selection.addRange(range)
  } else {
    const current = composerPlainText(editor)

    editor.append(
      buildRefFragment(parsed, {
        needsAfterSpace: true,
        needsBeforeSpace: current.length > 0 && !/\s$/.test(current)
      })
    )
    placeCaretEnd(editor)
  }

  normalizeComposerEditorDom(editor)

  return composerPlainText(editor)
}
