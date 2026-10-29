import type { ClipboardEvent, KeyboardEvent, MouseEvent } from 'react'
import { useRef } from 'react'

const TEXT_INPUT = 'input, textarea, select, [contenteditable="true"], [contenteditable=""]'
const INTERACTIVE = `a, button, ${TEXT_INPUT}`

/** Read views own selection without becoming editable or claiming focus on mount. */
export function usePreviewTextFocus(sourceText?: string) {
  const selectedAll = useRef<{ range: Range; text: string | undefined } | null>(null)

  return {
    'data-preview-text': '',
    role: 'document' as const,
    tabIndex: 0,
    onMouseDown: (event: MouseEvent<HTMLDivElement>) => {
      if (event.button !== 0) {
        return
      }

      selectedAll.current = null

      if (!(event.target as Element).closest(INTERACTIVE)) {
        event.currentTarget.focus({ preventScroll: true })
      }
    },
    onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => {
      if (
        event.defaultPrevented ||
        event.nativeEvent.isComposing ||
        event.key.toLowerCase() !== 'a' ||
        !(event.metaKey || event.ctrlKey) ||
        event.altKey ||
        event.shiftKey ||
        (event.target as Element).closest(TEXT_INPUT)
      ) {
        return
      }

      event.preventDefault()
      const selection = window.getSelection()
      const range = document.createRange()
      range.selectNodeContents(event.currentTarget)
      selection?.removeAllRanges()
      selection?.addRange(range)
      selectedAll.current = { range: range.cloneRange(), text: sourceText }
    },
    onCopy: (event: ClipboardEvent<HTMLDivElement>) => {
      const all = selectedAll.current
      const selection = window.getSelection()

      // Source is windowed: the native DOM range omits offscreen lines and includes
      // gutters. Only our intact select-all range copies the complete source buffer.
      if (all?.text === undefined || all.text !== sourceText || selection?.rangeCount !== 1) {
        return
      }

      const range = selection.getRangeAt(0)

      if (
        range.compareBoundaryPoints(Range.START_TO_START, all.range) ||
        range.compareBoundaryPoints(Range.END_TO_END, all.range)
      ) {
        return
      }

      event.preventDefault()
      event.clipboardData.setData('text/plain', all.text)
    },
    onBlur: () => {
      selectedAll.current = null
    }
  }
}
