import { useEffect, useState } from 'react'

// Tracks the app's dark/light mode off the `dark` class on <html> (set by
// themes/context.tsx). Embeds that theme their own content (tweets) read this.
export function useIsDark(): boolean {
  const [dark, setDark] = useState(
    () => typeof document !== 'undefined' && document.documentElement.classList.contains('dark')
  )

  useEffect(() => {
    if (typeof document === 'undefined') {
      return
    }

    const root = document.documentElement
    const observer = new MutationObserver(() => setDark(root.classList.contains('dark')))

    observer.observe(root, { attributeFilter: ['class'], attributes: true })

    return () => observer.disconnect()
  }, [])

  return dark
}
