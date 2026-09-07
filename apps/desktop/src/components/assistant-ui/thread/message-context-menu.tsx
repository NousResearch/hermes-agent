import { type ReactNode, useCallback, useEffect, useRef, useState } from 'react'

import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'
import { useComposerScope } from '@/app/chat/composer/scope'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuSeparator, ContextMenuTrigger } from '@/components/ui/context-menu'
import { writeClipboardText } from '@/components/ui/copy-button'
import { useI18n } from '@/i18n'
import { addComposerTextAttachment } from '@/store/composer'

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

  // A zero-offset range inside a text node reports intersectsNode() === false
  // on some engines; anchor/focus containment covers the degenerate cases.
  return (
    range.intersectsNode(host) ||
    host.contains(selection.anchorNode) ||
    host.contains(selection.focusNode)
  )
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
  const [hasSelection, setHasSelection] = useState(false)
  const snapshotRef = useRef<SelectionSnapshot>(EMPTY_SNAPSHOT)
  const hostRef = useRef<HTMLElement | null>(null)

  // Track the live selection so the trigger's `disabled` flag follows it —
  // but only selections that intersect THIS message. selectionchange fires on
  // drag-end, click-away, and programmatic clears — every path that changes
  // the selection. State (not a ref) so the flag is reactive, and the updater
  // bails out when the value hasn't actually flipped so steady-state
  // selection changes don't re-render at all.
  useEffect(() => {
    const sync = () => {
      const next = selectionIntersects(window.getSelection(), hostRef.current)
      setHasSelection(previous => (previous === next ? previous : next))
    }
    document.addEventListener('selectionchange', sync)

    return () => document.removeEventListener('selectionchange', sync)
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