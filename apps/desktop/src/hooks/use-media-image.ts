import { useStore } from '@nanostores/react'
import { type CSSProperties, useEffect, useMemo, useState } from 'react'

import { useComposerScope } from '@/app/chat/composer/scope'
import {
  getMediaImageDimensions,
  isInlineMediaSrc,
  isKnownBrokenMediaImage,
  type MediaImageDimensions,
  mediaImageKey,
  rememberMediaImageDimensions,
  rememberMediaImageFailure,
  resolveMediaDisplaySrc
} from '@/lib/media'
import { $connection } from '@/store/session'

type MediaImageResolver = (path: string) => Promise<string>

// Resolver identity is an authority boundary, not just an alternate transport.
// Weak keys do not retain disposed document loaders; the dimension cache stays bounded.
const resolverIds = new WeakMap<MediaImageResolver, number>()
let nextResolverId = 0

function resolverKey(resolveSrc?: MediaImageResolver) {
  if (!resolveSrc) {
    return ''
  }

  let id = resolverIds.get(resolveSrc)

  if (id === undefined) {
    id = ++nextResolverId
    resolverIds.set(resolveSrc, id)
  }

  return `resolver:${id}:`
}

/** Keep a frame for one source/owner for its mounted lifetime. Hints are not
 * intrinsic dimensions: a cold image is contained in that frame, not allowed
 * to resize it at decode. The next mount can use the measured dimensions.
 *
 * `preservePendingFrame` is for a caller whose content is all absolutely
 * positioned (a generated image): it always gets a frame, since without one a
 * successful retry renders into a zero-height box, and it keeps the frame
 * shown while its tool was pending. */
export function useMediaImage(
  path: string,
  fallbackRatio: number,
  intrinsic?: MediaImageDimensions,
  { preservePendingFrame = false, resolveSrc }: { preservePendingFrame?: boolean; resolveSrc?: MediaImageResolver } = {}
) {
  const connection = useStore($connection)
  const scope = useComposerScope()
  const connectionId = scope.connectionId || connection?.connectionId
  const profile = scope.profile ?? connection?.profile

  const owner = useMemo(
    () => (scope.connectionId || scope.profile ? { connectionId, profile } : undefined),
    [scope.connectionId, scope.profile, connectionId, profile]
  )

  const authority = resolverKey(resolveSrc)
  const key = authority + mediaImageKey(path, connection, owner)
  const ownerKey = authority + mediaImageKey('', connection, owner)

  const initialState = () => {
    const dimensions = intrinsic ?? getMediaImageDimensions(key)
    const ratio = dimensions ? dimensions.width / dimensions.height : fallbackRatio

    return {
      key,
      ownerKey,
      path,
      fallbackRatio,
      // A source that just failed gets no frame: it would only collapse again.
      frameStyle:
        !dimensions && !preservePendingFrame && isKnownBrokenMediaImage(key)
          ? undefined
          : ({
              aspectRatio: ratio,
              width: `min(calc(var(--image-preview-height) * ${ratio}), var(--image-preview-max-width), 100%${dimensions ? `, ${dimensions.width}px` : ''})`
            } satisfies CSSProperties),
      src: path && !resolveSrc && isInlineMediaSrc(path) ? path : '',
      loaded: false,
      failed: false
    }
  }

  const [state, setState] = useState(initialState)

  // Source changes must not paint the previous image/geometry even for one
  // commit. React retries this component before committing its children.
  // With no source yet, nothing is painted, so a hint arriving late reshapes.
  if (state.key !== key || (!path && state.fallbackRatio !== fallbackRatio)) {
    const next = initialState()
    // A generated result fills the frame already visible while its tool was
    // pending. Media state still resets; geometry is never inherited across
    // owners or between two non-empty sources.
    const inheritsPendingFrame = preservePendingFrame && state.ownerKey === ownerKey && !state.path && Boolean(path)

    setState(inheritsPendingFrame ? { ...next, frameStyle: state.frameStyle } : next)
  }

  useEffect(() => {
    let cancelled = false

    if (path && (resolveSrc || !isInlineMediaSrc(path))) {
      // A scoped resolver must never retry via the unscoped media transport.
      void (resolveSrc ? resolveSrc(path) : resolveMediaDisplaySrc(path, owner)).then(
        src => {
          if (!cancelled) {
            if (!src) {
              rememberMediaImageFailure(key)
            }

            setState(current => (current.key === key ? { ...current, src, failed: !src } : current))
          }
        },
        () => {
          if (!cancelled) {
            rememberMediaImageFailure(key)
            setState(current => (current.key === key ? { ...current, failed: true } : current))
          }
        }
      )
    }

    return () => {
      cancelled = true
    }
  }, [key, path, owner, resolveSrc])

  return {
    ...state,
    onLoad: (image: HTMLImageElement) => {
      rememberMediaImageDimensions(key, image.naturalWidth, image.naturalHeight)
      setState(current => ({ ...current, loaded: true }))
    },
    onError: () => {
      rememberMediaImageFailure(key)
      setState(current => ({ ...current, failed: true, loaded: false }))
    }
  }
}
