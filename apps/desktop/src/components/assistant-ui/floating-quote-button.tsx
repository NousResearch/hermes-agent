'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import { requestComposerInsert } from '@/app/chat/composer/focus'
import { type BrowserSelectionMatch, currentThreadSelection } from '@/app/chat/selection-reference'
import { isAddSelectionShortcut } from '@/app/right-sidebar/terminal/selection'
import { formatRefValue } from '@/components/assistant-ui/directive-text'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Quote as QuoteIcon } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { nextComposerContextReferenceLabel, setComposerSelectionReference } from '@/store/composer'

// Hide button after this many ms of no selection (avoids flicker when clicking)
const HIDE_DELAY_MS = 300
// Vertical distance from the top of the selection rect to the bottom of the button
const BUTTON_OFFSET = 8

export function FloatingQuoteButton() {
  const { t } = useI18n()
  const [visible, setVisible] = useState(false)
  const [position, setPosition] = useState<{ top: number; left: number }>({ top: 0, left: 0 })
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearHideTimer = useCallback(() => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current)
      hideTimerRef.current = null
    }
  }, [])

  const scheduleHide = useCallback(() => {
    clearHideTimer()
    hideTimerRef.current = setTimeout(() => setVisible(false), HIDE_DELAY_MS)
  }, [clearHideTimer])

  const quoteSelection = useCallback((match: BrowserSelectionMatch) => {
    const label = nextComposerContextReferenceLabel('selection', '_selection')

    setComposerSelectionReference(label, match.text)
    requestComposerInsert(`@selection:${formatRefValue(label)}`, { mode: 'inline', target: 'main' })
    triggerHaptic('selection')
    match.selection.removeAllRanges()
    setVisible(false)
  }, [])

  const handleSelectionChange = useCallback(() => {
    const match = currentThreadSelection()

    if (!match) {
      scheduleHide()

      return
    }

    clearHideTimer()

    // Position the button just above the selection, horizontally centered
    const rect = match.range.getBoundingClientRect()
    const viewportRect = match.viewport.getBoundingClientRect()

    const buttonHeight = 32
    const buttonHalfWidth = 16
    const top = rect.top - viewportRect.top - buttonHeight - BUTTON_OFFSET
    const left = rect.left + rect.width / 2 - viewportRect.left - buttonHalfWidth

    setPosition({
      top: Math.max(top, 4),
      left: Math.max(left, 4)
    })
    setVisible(true)
  }, [clearHideTimer, scheduleHide])

  const handleQuote = useCallback(() => {
    const match = currentThreadSelection()

    if (match) {
      quoteSelection(match)
    }
  }, [quoteSelection])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setVisible(false)

        return
      }

      if (!isAddSelectionShortcut(e)) {
        return
      }

      const match = currentThreadSelection()

      if (!match) {
        return
      }

      e.preventDefault()
      e.stopPropagation()
      quoteSelection(match)
    },
    [quoteSelection]
  )

  useEffect(() => {
    document.addEventListener('selectionchange', handleSelectionChange)
    document.addEventListener('keydown', handleKeyDown)

    return () => {
      document.removeEventListener('selectionchange', handleSelectionChange)
      document.removeEventListener('keydown', handleKeyDown)
      clearHideTimer()
    }
  }, [handleSelectionChange, handleKeyDown, clearHideTimer])

  if (!visible) {
    return null
  }

  return (
    <button
      aria-label={t.desktop.quoteSelection}
      className={cn(
        'pointer-events-auto absolute z-50 flex h-8 w-8 items-center justify-center',
        'rounded-md bg-popover text-muted-foreground shadow-lg',
        'ring-1 ring-(--ui-stroke-primary)',
        'transition-all duration-150 ease-out',
        'hover:bg-(--ui-row-hover-background) hover:text-foreground hover:scale-110',
        'active:scale-95'
      )}
      onClick={handleQuote}
      style={{ top: `${position.top}px`, left: `${position.left}px` }}
      title={t.desktop.quoteSelection}
      type="button"
    >
      <QuoteIcon className="size-4" />
    </button>
  )
}
