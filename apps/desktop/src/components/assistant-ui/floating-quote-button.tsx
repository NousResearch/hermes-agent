'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import { requestComposerInsert } from '@/app/chat/composer/focus'
import { useI18n } from '@/i18n'
import { Quote as QuoteIcon } from '@/lib/icons'
import { cn } from '@/lib/utils'

const VIEWPORT_SELECTOR = '[data-slot="aui_thread-viewport"]'
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

  const handleSelectionChange = useCallback(() => {
    const selection = window.getSelection()
    if (!selection || selection.isCollapsed || !selection.rangeCount) {
      scheduleHide()
      return
    }

    const selectedText = selection.toString().trim()
    if (!selectedText) {
      scheduleHide()
      return
    }

    // Require the selection to be inside the thread viewport
    const viewport = document.querySelector(VIEWPORT_SELECTOR)
    if (!viewport) {
      setVisible(false)
      return
    }

    const range = selection.getRangeAt(0)
    if (!viewport.contains(range.commonAncestorContainer)) {
      setVisible(false)
      return
    }

    clearHideTimer()

    // Position the button just above the selection, horizontally centered
    const rect = range.getBoundingClientRect()
    const viewportRect = viewport.getBoundingClientRect()

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
    const selection = window.getSelection()
    if (!selection) return

    const text = selection.toString().trim()
    if (!text) return

    requestComposerInsert(text, { mode: 'block' })
    selection.removeAllRanges()
    setVisible(false)
  }, [])

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      setVisible(false)
    }
  }, [])

  useEffect(() => {
    document.addEventListener('selectionchange', handleSelectionChange)
    document.addEventListener('keydown', handleKeyDown)

    return () => {
      document.removeEventListener('selectionchange', handleSelectionChange)
      document.removeEventListener('keydown', handleKeyDown)
      clearHideTimer()
    }
  }, [handleSelectionChange, handleKeyDown, clearHideTimer])

  if (!visible) return null

  return (
    <button
      aria-label={t.desktop.quoteSelection}
      className={cn(
        'pointer-events-auto absolute z-50 flex h-8 w-8 items-center justify-center',
        'rounded-md bg-(--ui-popover-background) text-muted-foreground shadow-lg',
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
