import { type ComponentType, lazy } from 'react'

export interface LazyView<P extends object> {
  (props: P): React.JSX.Element
  /** Fetch + evaluate the chunk now; a later render then draws synchronously. */
  preload: () => Promise<ComponentType<P>>
}

/**
 * A code-split view that renders synchronously once its chunk is in.
 *
 * `React.lazy` suspends on its first render even when the chunk is already cached, and React
 * holds a Suspense reveal for 300ms after its fallback commits, so every first visit to a lazy
 * page sat on a blank frame for at least that long. Here a preloaded (or once-rendered) view
 * renders its component directly; only a genuinely missing chunk goes through `React.lazy`.
 */
export function lazyView<P extends object>(load: () => Promise<ComponentType<P>>): LazyView<P> {
  let loaded: ComponentType<P> | undefined
  let pending: Promise<ComponentType<P>> | undefined

  const preload = () =>
    (pending ??= load().then(
      component => (loaded = component),
      (error: unknown) => {
        // A failed warm-up (offline Webapp, replaced build) must not poison the next attempt.
        pending = undefined

        throw error
      }
    ))

  // Resolves only after `preload` has set `loaded`, so a retry after the suspension renders the
  // loaded component itself: the lazy wrapper never commits, and the page never changes type
  // (which would remount it) once its chunk lands.
  const Suspending = lazy(async () => ({ default: await preload() }))

  function LazyViewComponent(props: P) {
    // `loaded` is module state the compiler cannot see change; a memoized element could pin the
    // suspending wrapper past the chunk's arrival.
    'use no memo'

    const View = loaded ?? Suspending

    return <View {...props} />
  }

  return Object.assign(LazyViewComponent, { preload })
}
