import { useCallback, useMemo, useRef, useState } from 'react'

export function useStickToBottom() {
  const scrollRef = useRef<HTMLElement | null>(null)
  const contentRef = useRef<HTMLElement | null>(null)
  const [isAtBottom, setIsAtBottom] = useState(true)

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current
    if (el) {
      el.scrollTop = el.scrollHeight
    }
    setIsAtBottom(true)
  }, [])

  const stopScroll = useCallback(() => {
    setIsAtBottom(false)
  }, [])

  return useMemo(
    () => ({
      contentRef,
      isAtBottom,
      scrollRef,
      scrollToBottom,
      stopScroll,
    }),
    [isAtBottom, scrollToBottom, stopScroll]
  )
}

export const StickToBottom = {
  Content({ children, ...props }: any) {
    return <div {...props}>{children}</div>
  },
}
