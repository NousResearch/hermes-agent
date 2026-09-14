import { formatRefValue } from '@/components/assistant-ui/directive-text'
import { translateNow } from '@/i18n'
import { contextPath } from '@/lib/chat-runtime'

import type { DroppedFile } from '../hooks/use-composer-actions'

import {
  composerPlainText,
  normalizeComposerEditorDom,
  placeCaretEnd,
  refChipElement,
  RICH_INPUT_SLOT
} from './rich-editor'

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

/**
 * Windows Explorer / Chromium sparse external file drag.
 *
 * During `dragenter`/`dragover` the OS can expose an entirely empty
 * `DataTransfer` (no `types`, no `items` with `kind=file`) while the `File`
 * payload only materialises at `drop`. The ONLY known production source of
 * such a fully empty native DataTransfer on this app is an external OS file
 * drag — internal in-app drags always carry at least `text/plain` or the
 * `application/x-hermes-paths` custom MIME. An empty dragenter is therefore
 * a credible signal of an external file drag, even though it is not yet
 * proof of one. Callers that use this for `preventDefault` MUST still
 * verify `extractDroppedFiles(...).length > 0` at drop time.
 *
 * This intentionally does NOT consider transfers with a non-empty `types`
 * array: those are either real file drags (`Files`), in-app drags
 * (`application/x-hermes-paths`), or unrelated text drags (`text/plain`) —
 * the existing `dragHasAttachments` helper already classifies them.
 */
export function isSparseExternalFileDrag(transfer: DataTransfer | null): boolean {
  if (!transfer) {
    return false
  }

  const types = transfer.types

  if (types && types.length !== 0) {
    return false
  }

  const items = transfer.items

  if (items && items.length !== 0) {
    return false
  }

  return true
}

/**
 * Stateful, event-lifecycle-aware drag accept gate. Use this to wrap a
 * native HTML5 drop zone so the bubble-phase `dragover` keeps drop alive
 * for the Windows Explorer sparse case (see {@link isSparseExternalFileDrag})
 * WITHOUT blanket-classifying every empty `DataTransfer` as a file drag.
 *
 * Invariants:
 * - `onEnter` arms the gate on the FIRST credible signal (typed file drag
 *   OR a fully-empty transfer). Once armed, subsequent `onOver` calls
 *   accept the drag even if the transfer is still empty.
 * - A `dragover` that fires WITHOUT a prior `dragenter` for this drag does
 *   NOT arm the gate and is rejected — no spurious "file drag active" UI.
 * - `onLeave(atRootDepth: true)` and `reset()` clear the gate. The caller
 *   tracks nested dragenter/leave depth and only calls `onLeave(true)` when
 *   depth returns to 0.
 * - This is intentionally NOT a React hook so the gate can be unit-tested
 *   without a renderer; callers keep it in a `useRef` for stability.
 */
export interface DragLifecycleGate {
  /** Returns true if the dragenter should be accepted (and preventDefault'd). */
  onEnter(transfer: DataTransfer | null): boolean
  /** Returns true if the dragover should keep drop alive (and preventDefault'd). */
  onOver(transfer: DataTransfer | null): boolean
  /** Caller passes `true` when leaving the root depth to disarm the gate. */
  onLeave(atRootDepth: boolean): void
  /** Force-clear the gate (drop, unmount, or explicit abort). */
  reset(): void
}

export function createDragLifecycleGate(pathsMime: string): DragLifecycleGate {
  let crediblyArmed = false

  return {
    onEnter(transfer) {
      if (dragHasAttachments(transfer, pathsMime)) {
        crediblyArmed = true

        return true
      }

      if (isSparseExternalFileDrag(transfer)) {
        // Tentative arm: an empty transfer is the only known signal of a
        // Windows Explorer / Chromium sparse file drag. The drop handler
        // must still gate on `extractDroppedFiles(...).length > 0`.
        crediblyArmed = true

        return true
      }

      return false
    },

    onOver(transfer) {
      if (dragHasAttachments(transfer, pathsMime)) {
        return true
      }

      // Stay open for the same drag we tentatively armed on enter — the
      // browser keeps re-firing dragover until drop or dragleave, and
      // dropping the gate here would let the app-lifetime HTML5Backend's
      // capture-time `dropEffect='none'` suppress the eventual drop.
      return crediblyArmed
    },

    onLeave(atRootDepth) {
      if (atRootDepth) {
        crediblyArmed = false
      }
    },

    reset() {
      crediblyArmed = false
    }
  }
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

  // Carry the editor's slot marker: composerPlainText appends a trailing "\n"
  // to any other block element, so a bare <div> made `beforeText` always look
  // like it ended in whitespace and the separating space was never inserted —
  // a chip dropped after a word came out glued to it (`review@file:...`).
  const container = document.createElement('div')
  container.dataset.slot = RICH_INPUT_SLOT
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

export function insertInlineRefsIntoEditor(
  editor: HTMLDivElement,
  refs: readonly InlineRefInput[],
  { interactive = true }: { interactive?: boolean } = {}
) {
  const parsed = refs.map(parseInlineRef).filter((ref): ref is NonNullable<typeof ref> => ref !== null)
  const hasEmptySentinel = editor.childNodes.length === 1 && editor.firstChild?.nodeName === 'BR'

  if (!parsed.length) {
    return null
  }

  if (hasEmptySentinel) {
    editor.replaceChildren()
  }

  if (interactive) {
    editor.focus({ preventScroll: true })
  }

  const selection = interactive ? window.getSelection() : null

  const range =
    !hasEmptySentinel && selection?.rangeCount && editor.contains(selection.getRangeAt(0).commonAncestorContainer)
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

    if (interactive) {
      placeCaretEnd(editor)
    }
  }

  normalizeComposerEditorDom(editor)

  return composerPlainText(editor)
}
