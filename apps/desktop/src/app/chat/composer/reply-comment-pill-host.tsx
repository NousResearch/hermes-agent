import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { MessageSquareText } from '@/lib/icons'
import { notify } from '@/store/notifications'

import { requestComposerFocus } from './focus'
import { createReplyComment, MAX_REPLY_COMMENTS } from './reply-comments'
import { ReplyCommentPopover, type ReplyCommentPopoverPosition } from './reply-comment-popover'
import { useComposerScope } from './scope'

/** Message bubbles a reply comment can anchor to. */
const MESSAGE_ROOT = '[data-slot="aui_assistant-message-content"],[data-slot="aui_user-message-root"]'
/** Surfaces whose own selection must never arm the pill. */
const EDITABLE_ROOT =
  '[data-slot="composer-rich-input"],[data-slot="aui_edit-composer-root"],input,textarea,[contenteditable="true"]'

const MIN_SELECTION_CHARS = 2

interface PillState extends ReplyCommentPopoverPosition {
  quote: string
}

/**
 * Floating "Comment" pill for transcript selections. Select text in an
 * assistant or user message (mouse or right-click) → pill → note popover →
 * Enter pins a chip above the composer. Several chips batch into one send.
 *
 * Mounted once per ChatView inside the composer scope provider, so tiles pin
 * to their own composer and the main chat to its own.
 */
export function ReplyCommentPillHost() {
  const { t } = useI18n()
  const scope = useComposerScope()
  const copy = t.composer.replyComments
  const [pill, setPill] = useState<PillState | null>(null)
  const [popover, setPopover] = useState<(PillState & { key: number }) | null>(null)

  useEffect(() => {
    const selectionInMessage = (): { quote: string; rect: DOMRect } | null => {
      const selection = window.getSelection()

      if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
        return null
      }

      const quote = selection.toString().replace(/\s+/g, ' ').trim()

      if (quote.length < MIN_SELECTION_CHARS) {
        return null
      }

      const anchorEl =
        selection.anchorNode instanceof Element ? selection.anchorNode : selection.anchorNode?.parentElement
      const focusEl = selection.focusNode instanceof Element ? selection.focusNode : selection.focusNode?.parentElement

      if (!anchorEl || !focusEl || anchorEl.closest(EDITABLE_ROOT) || focusEl.closest(EDITABLE_ROOT)) {
        return null
      }

      const root = anchorEl.closest(MESSAGE_ROOT)

      if (!root || !root.contains(focusEl)) {
        return null
      }

      return { quote, rect: selection.getRangeAt(0).getBoundingClientRect() }
    }

    const placePill = (quote: string, x: number, y: number) => {
      setPopover(null)
      setPill({ quote, x: Math.max(8, x - 48), y: Math.max(8, y - 44) })
    }

    const onMouseUp = () => {
      // Let the click that opens the popover (or a chip button) land first —
      // re-arming on its mouseup would resurrect the pill under the popover.
      if (popover) {
        return
      }

      const hit = selectionInMessage()

      if (!hit) {
        setPill(null)
        return
      }

      placePill(hit.quote, hit.rect.left + hit.rect.width / 2, hit.rect.top)
    }

    const onContextMenu = (event: MouseEvent) => {
      if (popover) {
        return
      }

      const hit = selectionInMessage()

      if (!hit) {
        setPill(null)
        return
      }

      // Right-click with a live selection: offer the pill at the cursor. The
      // native menu still opens — the pill is the faster path, not a hijack.
      placePill(hit.quote, event.clientX, event.clientY)
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setPill(null)
        setPopover(null)
      }
    }

    const hideOnScroll = () => {
      setPill(null)
    }

    document.addEventListener('mouseup', onMouseUp)
    document.addEventListener('contextmenu', onContextMenu)
    document.addEventListener('keydown', onKeyDown, true)
    document.addEventListener('scroll', hideOnScroll, true)

    return () => {
      document.removeEventListener('mouseup', onMouseUp)
      document.removeEventListener('contextmenu', onContextMenu)
      document.removeEventListener('keydown', onKeyDown, true)
      document.removeEventListener('scroll', hideOnScroll, true)
    }
  }, [popover])

  const commit = (quote: string, note: string) => {
    const comment = createReplyComment(quote, note)

    if (!comment) {
      return
    }

    if (!scope.replyComments.add(comment)) {
      notify({ kind: 'warning', message: copy.limitReached(MAX_REPLY_COMMENTS), title: copy.comment })
      return
    }

    requestComposerFocus(scope.target)
    window.getSelection()?.removeAllRanges()
    setPopover(null)
    setPill(null)
  }

  return (
    <>
      {pill &&
        !popover &&
        createPortal(
          <div className="fixed z-[70]" data-slot="reply-comment-pill" style={{ left: pill.x, top: pill.y }}>
            <Button
              className="h-7 gap-1 rounded-md px-2 text-[0.72rem] shadow-md backdrop-blur-md"
              onMouseDown={event => {
                // Keep the transcript selection alive: without this the
                // mousedown collapses it before the click commits the quote.
                event.preventDefault()
                event.stopPropagation()
              }}
              onClick={event => {
                event.stopPropagation()
                setPopover({ key: Date.now(), quote: pill.quote, x: pill.x, y: pill.y })
              }}
              type="button"
              variant="secondary"
            >
              <MessageSquareText />
              {copy.comment}
            </Button>
          </div>,
          document.body
        )}
      {popover && (
        <ReplyCommentPopover
          confirmLabel={copy.attach}
          initialNote=""
          key={popover.key}
          onClose={() => setPopover(null)}
          onConfirm={note => commit(popover.quote, note)}
          position={popover}
          quote={popover.quote}
        />
      )}
    </>
  )
}
