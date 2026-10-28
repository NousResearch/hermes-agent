import { useCallback, useEffect, useRef, useState } from 'react'

/** Keep plaintext local to the visible credential surface, never its cache. */
export function useEphemeralReveal(scope?: string) {
  const [revealed, setRevealed] = useState<Record<string, string>>({})
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>())
  const requests = useRef(new Map<string, symbol>())
  const mounted = useRef(false)

  const hide = useCallback((key: string) => {
    clearTimeout(timers.current.get(key))
    timers.current.delete(key)
    requests.current.delete(key)
    setRevealed(previous => {
      const next = { ...previous }
      delete next[key]
      return next
    })
  }, [])

  const clear = useCallback(() => {
    timers.current.forEach(clearTimeout)
    timers.current.clear()
    requests.current.clear()
    setRevealed({})
  }, [])

  const reveal = useCallback(async (key: string, load: () => Promise<string>) => {
    clearTimeout(timers.current.get(key))
    timers.current.delete(key)
    const request = Symbol()
    requests.current.set(key, request)
    const value = await load()

    // Leaving, hiding, or switching profiles also invalidates in-flight reveals.
    if (!mounted.current || document.hidden || requests.current.get(key) !== request) {
      return
    }

    clearTimeout(timers.current.get(key))
    setRevealed(previous => ({ ...previous, [key]: value }))
    timers.current.set(key, setTimeout(() => hide(key), 30_000))
  }, [hide])

  // eslint-disable-next-line no-restricted-syntax -- mount flag guarding async reveal completion, not an atom mirror
  useEffect(() => {
    mounted.current = true
    const onVisibilityChange = () => {
      if (document.hidden) {
        clear()
      }
    }

    document.addEventListener('visibilitychange', onVisibilityChange)
    window.addEventListener('blur', clear)

    return () => {
      mounted.current = false
      document.removeEventListener('visibilitychange', onVisibilityChange)
      window.removeEventListener('blur', clear)
      clear()
    }
  }, [scope, clear])

  return { revealed, reveal, hide, clear }
}
