import { type FC, useCallback, useEffect, useRef, useState } from 'react'

import { useComposerScope } from '@/app/chat/composer/scope'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { formatMessageTimestamp, quoteMessageIntoComposer, type QuoteRole } from '@/lib/quote-message'

interface QuoteTarget {
  below: boolean
  role: QuoteRole
  text: string
  x: number
  y: number
}

/** One control per thread; only the exact selection in one message is quoted. */
export const QuoteSelectionAction: FC = () => {
  const { t } = useI18n()
  const scope = useComposerScope()
  const [target, setTarget] = useState<QuoteTarget | null>(null)
  const originRef = useRef<HTMLSpanElement | null>(null)

  const clear = useCallback(() => setTarget(null), [])

  const evaluate = useCallback(() => {
    const selection = window.getSelection()
    const container = originRef.current?.parentElement

    if (!selection || selection.isCollapsed || !selection.rangeCount || !container) {
      clear()

      return
    }

    const text = selection.toString()
    const range = selection.getRangeAt(0)
    const start = range.startContainer instanceof Element ? range.startContainer : range.startContainer.parentElement
    const end = range.endContainer instanceof Element ? range.endContainer : range.endContainer.parentElement
    const selector = '[data-role="assistant"], [data-role="user"]'
    const bubble = start?.closest(selector)

    // Each mounted thread listens globally; only the selection's owner paints.
    if (
      !text.trim() ||
      !bubble ||
      bubble !== end?.closest(selector) ||
      !container.contains(bubble) ||
      start?.closest('[contenteditable="true"], input, textarea') ||
      end?.closest('[contenteditable="true"], input, textarea')
    ) {
      clear()

      return
    }

    const rect = range.getBoundingClientRect()
    const origin = container.getBoundingClientRect()

    if (!rect.width && !rect.height) {
      clear()

      return
    }

    const below = rect.top - origin.top < 48
    const x = Math.max(56, Math.min(rect.left + rect.width / 2 - origin.left, origin.width - 56))
    const y = below ? rect.bottom - origin.top + 8 : rect.top - origin.top - 8
    setTarget({ below, role: bubble.getAttribute('data-role') as QuoteRole, text, x, y })
  }, [clear])

  useEffect(() => {
    let frame = 0

    const schedule = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(evaluate)
    }

    document.addEventListener('selectionchange', schedule)
    document.addEventListener('pointerup', schedule)

    return () => {
      cancelAnimationFrame(frame)
      document.removeEventListener('selectionchange', schedule)
      document.removeEventListener('pointerup', schedule)
    }
  }, [evaluate])

  useEffect(() => {
    if (!target) {
      return
    }

    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        clear()
      }
    }

    window.addEventListener('scroll', clear, true)
    window.addEventListener('wheel', clear, { capture: true, passive: true })
    window.addEventListener('resize', clear)
    document.addEventListener('keydown', onKey)

    return () => {
      window.removeEventListener('scroll', clear, true)
      window.removeEventListener('wheel', clear, true)
      window.removeEventListener('resize', clear)
      document.removeEventListener('keydown', onKey)
    }
  }, [clear, target])

  return (
    <>
      <span aria-hidden className="pointer-events-none absolute left-0 top-0 size-0" ref={originRef} />
      {target && (
        <Button
          aria-label={t.assistant.thread.quoteMessage}
          className="absolute z-40"
          data-slot="selection-quote-action"
          onClick={() => {
            quoteMessageIntoComposer(target.role, target.text, formatMessageTimestamp(), scope.target)
            window.getSelection()?.removeAllRanges()
            clear()
          }}
          onMouseDown={event => event.preventDefault()}
          onPointerDown={event => event.preventDefault()}
          size="sm"
          style={{
            left: target.x,
            top: target.y,
            transform: target.below ? 'translate(-50%, 0)' : 'translate(-50%, -100%)'
          }}
          type="button"
          variant="floating"
        >
          <Codicon name="comment-quote" />
          {t.assistant.thread.quoteMessage}
        </Button>
      )}
    </>
  )
}
