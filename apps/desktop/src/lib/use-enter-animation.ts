import { useCallback, useRef } from 'react'

/**
 * One-shot enter animation via the Web Animations API.
 *
 * Returns a callback ref. The animation fires exactly once when the element
 * first attaches to the DOM and never replays for an already-mounted node —
 * this is deliberate. CSS-transition + `@starting-style` is fragile here
 * because:
 *   - Streaming deltas constantly invalidate ancestor state, which can
 *     re-trigger transitions on unrelated descendants.
 *   - `@starting-style` only covers DOM insertion / first-match, but any
 *     style restart during the message lifecycle replays the transition.
 *   - Some Chromium versions reset transitions when an attribute on an
 *     ancestor toggles, even if the descendant's properties never change.
 *
 * `el.animate(...)` runs against the element directly and is independent of
 * CSS rule churn — it plays once, finishes, and is done. If the element
 * unmounts and re-mounts, the callback ref runs again and replays it
 * (correct behaviour).
 *
 * `enabled` is captured at mount-time only — flipping it later doesn't
 * suddenly play the animation on existing nodes.
 */
const playedAnimationKeys = new Set<string>()
const playedAnimationOrder: string[] = []
const MAX_TRACKED_KEYS = 2048

function hasPlayedAnimation(key: string): boolean {
  return playedAnimationKeys.has(key)
}

function rememberPlayedAnimation(key: string): void {
  if (playedAnimationKeys.has(key)) {
    return
  }

  playedAnimationKeys.add(key)
  playedAnimationOrder.push(key)

  if (playedAnimationOrder.length > MAX_TRACKED_KEYS) {
    const evicted = playedAnimationOrder.shift()

    if (evicted) {
      playedAnimationKeys.delete(evicted)
    }
  }
}

function scheduleMicrotask(cb: () => void): void {
  if (typeof queueMicrotask === 'function') {
    queueMicrotask(cb)

    return
  }

  void Promise.resolve().then(cb)
}

export function useEnterAnimation(enabled: boolean, animationKey?: string): (el: HTMLElement | null) => void {
  const enabledRef = useRef(enabled)
  const keyRef = useRef(animationKey)

  enabledRef.current = enabled
  keyRef.current = animationKey

  return useCallback((el: HTMLElement | null) => {
    if (!el || !enabledRef.current || typeof window === 'undefined') {
      return
    }

    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      return
    }

    const key = keyRef.current

    if (key && hasPlayedAnimation(key)) {
      return
    }

    el.animate(
      [
        { opacity: 0, transform: 'translateY(0.5rem)' },
        { opacity: 1, transform: 'translateY(0)' }
      ],
      { duration: 220, easing: 'linear', fill: 'both' }
    )

    if (key) {
      // In React StrictMode the first mount can be immediately torn down.
      // Only persist "played" once the element survives to the microtask tick.
      scheduleMicrotask(() => {
        if (el.isConnected) {
          rememberPlayedAnimation(key)
        }
      })
    }
  }, [])
}
