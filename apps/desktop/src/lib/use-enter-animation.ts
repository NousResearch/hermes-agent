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

    // The Web Animations document timeline PAUSES while the tab/window is
    // hidden (alt-tab, app switch, OS occlusion). An enter animation with
    // `fill: 'both'` created — or frozen mid-flight — while hidden pins the
    // element at its opening keyframe (`opacity: 0`) while it still occupies
    // full layout height, so on return the user sees a blank gap where the
    // message/reasoning text should be. Guard BOTH ends: skip entirely when
    // already hidden, and force-finish (jump to the fully-visible end frame)
    // the instant the document goes hidden mid-play. `finish()` sets
    // currentTime to the end synchronously regardless of timeline state, so
    // the computed style resolves to the end keyframe (opacity: 1) even while
    // paused — the element can never be stranded transparent.
    if (typeof document !== 'undefined' && document.hidden) {
      if (key) {
        rememberPlayedAnimation(key)
      }

      return
    }

    const animation = el.animate(
      [
        { opacity: 0, transform: 'translateY(0.375rem)' },
        { opacity: 1, transform: 'translateY(0)' }
      ],
      { duration: 180, easing: 'cubic-bezier(0.16, 1, 0.3, 1)', fill: 'both' }
    )

    if (typeof document !== 'undefined' && typeof animation.finished?.then === 'function') {
      const finishIfHidden = () => {
        if (document.hidden) {
          try {
            animation.finish()
          } catch {
            // Already finished/cancelled — nothing to settle.
          }
        }
      }

      document.addEventListener('visibilitychange', finishIfHidden)

      const cleanup = () => document.removeEventListener('visibilitychange', finishIfHidden)

      // Resolves on natural completion (180ms) OR an early finish()/cancel —
      // detaches the listener so we never accumulate one per message.
      animation.finished.then(cleanup, cleanup)
    }

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
