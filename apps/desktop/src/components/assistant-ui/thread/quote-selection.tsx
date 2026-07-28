import type { MouseEvent, ReactElement } from 'react'
import { useCallback, useRef } from 'react'

import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { useI18n } from '@/i18n'

export function quoteSelectedText(text: string): string {
  return text
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map(line => `> ${line}`)
    .join('\n')
}

export function selectedTextWithin(
  target: Node,
  selection = typeof window === 'undefined' ? null : window.getSelection()
): string {
  const range = selection?.rangeCount ? selection.getRangeAt(0) : null
  const text = selection?.toString() ?? ''

  return text.trim() && range && target.contains(range.commonAncestorContainer) ? text : ''
}

export function QuoteSelectionContextMenu({ children }: { children: ReactElement }) {
  const { t } = useI18n()
  const selectedTextRef = useRef('')

  const handleContextMenu = useCallback((event: MouseEvent) => {
    const text = selectedTextWithin(event.currentTarget)

    if (!text) {
      selectedTextRef.current = ''
      event.preventDefault()
      return
    }

    selectedTextRef.current = text
  }, [])

  const quote = useCallback(() => {
    const text = selectedTextRef.current

    if (!text.trim()) {
      return
    }

    requestComposerInsert(quoteSelectedText(text), { mode: 'block', target: 'active' })
    requestComposerFocus('active')
  }, [])

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild onContextMenu={handleContextMenu}>
        {children}
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={quote}>{t.assistant.thread.quoteInNewMessage}</ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}
