import { type ReactNode, useCallback, useEffect, useRef, useState } from 'react'

import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'
import { useComposerScope } from '@/app/chat/composer/scope'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuSeparator, ContextMenuTrigger } from '@/components/ui/context-menu'
import { writeClipboardText } from '@/components/ui/copy-button'
import { useI18n } from '@/i18n'
import { addComposerTextAttachment } from '@/store/composer'
import { subscribeToDocumentSelection } from '@/store/selection'

interface MessageContextMenuProps {
  children: ReactNode
  /** The message element to attach the context menu to. */
  messageId?: string
}

interface SelectionSnapshot {
  text: string
  anchor: Element | null
}

const EMPTY_SNAPSHOT: SelectionSnapshot = { text: '', anchor: null }

/** True when the live selection is a real highlight that intersects this
 *  message's own subtree — not merely somewhere in the document. Selecting in
 *  the composer (or another message) must not arm THIS message's menu, or a
 *  right-click here would stage foreign text and Select All would silently
 *  no-op (its container lookup keys off the selection's anchor). */
function selectionIntersects(selection: Selection | null, host: Element | null): boolean {
  if (!selection || selection.isCollapsed || host == null) {
    return false
  }

  // A range that ends exactly at the host's boundary reports empty text while
  // still "intersecting" — require actual selected text too, matching the
  // original `selection.toString().trim().length > 0` gate.
  if (!selection.toString().trim()) {
    return false
  }

  const range = selection.getRangeAt(0)

  // jsdom has no Range.intersectsNode; guard it and fall back to
  // anchor/focus containment. Cover both shapes: `selectNodeContents(host)`
  // makes the anchor/focus the host ELEMENT itself (not a child), and a
  // text-level selection anchors a child text node. Either satisfies us.
  const intersects =
    typeof range.intersectsNode === 'function'
      ? range.intersectsNode(host)
      : false

  return intersects ||
    host === (selection.anchorNode as Element) ||
    host === (selection.focusNode as Element) ||
    host.contains(selection.anchorNode) ||
    host.contains(selection.focusNode)
}

/** Shared right-click context menu for message blocks (user + assistant).
 *  When text is selected, the standard Copy / Select All actions stay,
 *  with "Add as context" (stages a chip) and "Paste as text" (inserts
 *  quoted text into the composer) added below — an addition, not a
 *  replacement.
 *
 *  The Radix wrapper is ALWAYS mounted and only the trigger's `disabled`
 *  flag follows the selection. Swapping the tree in and out on every
 *  selectionchange (the earlier design) unmounted the DOM nodes under the
 *  cursor mid-drag, which trashed the live browser selection (highlight
 *  jumped to a big unrelated area) and caused layout flicker. A stable
 *  tree with a disabled trigger lets the app context menu serve plain
 *  right-clicks when nothing is selected — without ever remounting the
 *  children. */
export function MessageContextMenu({ children, messageId }: MessageContextMenuProps) {
  const { t } = useI18n()
  const scope = useComposerScope()
  // Lazy-init from the CURRENT selection: a row that mounts while a selection
  // already exists over it (a new streaming row, a branch-restore remount)
  // must arm immediately — `selectionchange` only fires on the NEXT change,
  // and a disabled trigger in that window means no menu at all on a user
  // bubble (the app menu defers to the disabled trigger's subtree).
  const [hasSelection, setHasSelection] = useState(() => {
    // Lazy-init from the CURRENT document selection; the host ref doesn't
    // exist yet at mount, so "selection exists" is the honest initial state
    // — the effect's post-ref-mount check narrows it to this message.
    const selection = typeof document !== 'undefined' ? document.getSelection() : null

    return Boolean(selection && !selection.isCollapsed && selection.toString().trim().length > 0)
  })
  const snapshotRef = useRef<SelectionSnapshot>(EMPTY_SNAPSHOT)
  const hostRef = useRef<HTMLElement | null>(null)

  // Follow the shared document-selection atom (one listener for the whole
  // app) and intersect the selection with THIS message's subtree — a caret
  // move in the composer must not arm every row's trigger. The updater bails
  // out when the answer hasn't flipped so steady-state changes don't
  // re-render.
  useEffect(() => {
    // The lazy init runs before the ref exists (mount-time host is null), so
    // re-check once the element is attached.
    setHasSelection(selectionIntersects(window.getSelection(), hostRef.current))

    return subscribeToDocumentSelection(selection => {
      const next = selectionIntersects(selection, hostRef.current)
      setHasSelection(previous => (previous === next ? previous : next))
    })
  }, [])

  // Snapshot the selection at right-click time — before the menu takes
  // focus and could clear or move it. The trigger composes this handler
  // ahead of Radix's own, so it runs first in every mode.
  const captureSnapshot = useCallback((event: React.MouseEvent) => {
    const selection = window.getSelection()
    const text = selection?.toString().trim() ?? ''

    if (!text) {
      snapshotRef.current = EMPTY_SNAPSHOT

      return
    }

    const anchor = selection!.anchorNode

    snapshotRef.current = {
      text,
      anchor: anchor?.nodeType === 1 ? (anchor as Element) : (anchor?.parentElement ?? null)
    }
  }, [])

  const handleCopy = useCallback(() => {
    if (!snapshotRef.current.text) {
      return
    }

    void writeClipboardText(snapshotRef.current.text)
  }, [])

  const handleSelectAll = useCallback(() => {
    const selection = window.getSelection()
    const anchor = snapshotRef.current.anchor

    if (!selection || !anchor) {
      return
    }

    // User bubbles carry the message text in `.composer-human-message`;
    // assistant content sits under the aui content slot.
    const container =
      anchor.closest<HTMLElement>('.composer-human-message') ??
      anchor.closest<HTMLElement>('[data-slot="aui_assistant-message-content"]')

    if (!container) {
      return
    }

    const range = document.createRange()
    range.selectNodeContents(container)
    selection.removeAllRanges()
    selection.addRange(range)
  }, [])

  const handleAddAsContext = useCallback(() => {
    if (!snapshotRef.current.text) {
      return
    }

    // Route to the composer this message's surface owns: inside a session
    // tile that is the TILE's composer (its own chips, its own submit), not
    // the main chat's. Sibling gestures (drag-drop attach, shift+click)
    // already route per-scope the same way.
    addComposerTextAttachment(snapshotRef.current.text, messageId, scope.attachments)
    requestComposerFocus(scope.target)
  }, [messageId, scope])

  const handlePasteAsText = useCallback(() => {
    const text = snapshotRef.current.text

    if (!text) {
      return
    }

    const quoted = text
      .split('\n')
      .map(line => `> ${line}`)
      .join('\n')

    requestComposerInsert(quoted + '\n\n', { mode: 'block', target: scope.target })
  }, [scope])

  return (
    <ContextMenu>
      <ContextMenuTrigger
        asChild
        disabled={!hasSelection}
        onContextMenu={captureSnapshot}
        ref={(element: HTMLElement | null) => {
          hostRef.current = element
        }}
      >
        {children}
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={handleCopy}>{t.common.copy}</ContextMenuItem>
        <ContextMenuItem onSelect={handleSelectAll}>{t.common.selectAll}</ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={handleAddAsContext}>{t.common.addAsContext}</ContextMenuItem>
        <ContextMenuItem onSelect={handlePasteAsText}>{t.common.pasteAsText}</ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}