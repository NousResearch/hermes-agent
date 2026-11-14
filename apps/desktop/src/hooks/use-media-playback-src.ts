import { useStore } from '@nanostores/react'
import { type SyntheticEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useComposerScope } from '@/app/chat/composer/scope'
import { useSessionView } from '@/app/chat/session-view'
import { mediaKind, resolveMediaPlaybackSrc } from '@/lib/media'
import { cachedPlaybackSrc, forgetPlaybackSrc, rememberPlaybackSrc } from '@/lib/playback-source-cache'
import { $connection } from '@/store/session'

type PlaybackEvent = SyntheticEvent<HTMLAudioElement | HTMLVideoElement>

interface ResumePoint {
  element: HTMLAudioElement | HTMLVideoElement
  paused: boolean
  src: string
  time: number
}

/** The playable source for a transcript audio/video attachment.
 *
 * Webapp sources are stream-ticket URLs that expire, and the element needs a
 * valid one for every later range request (a seek, or play after a long
 * pause). A video fetches nothing until it is played, so its first ticket can
 * expire unused. So a source that has loaded, or that waited unfetched, is
 * re-resolved on its next error and playback resumes where it stopped. Any
 * other source that fails before it loads, or one that resolves to the same
 * URL again, is a real failure. */
export function useMediaPlaybackSrc(path: string) {
  const kind = mediaKind(path)
  const scope = useComposerScope()

  // A transcript in a background tile streams through the tile's owner, not the foreground.
  const owner = useMemo(
    () => (scope.connectionId || scope.profile ? { connectionId: scope.connectionId, profile: scope.profile } : undefined),
    [scope.connectionId, scope.profile]
  )

  // The transcript's own session: a Docker agent's /workspace clip lives in its sandbox.
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const runtimeId = useStore(view.$runtimeId)
  const sessionId = storedId || runtimeId || undefined

  // A remount reuses the URL an earlier mount resolved: the browser keeps what it fetched, and the
  // clip's remembered position (keyed by URL) still applies.
  const conn = useStore($connection)

  const cacheKey = JSON.stringify([
    path,
    owner?.connectionId ?? conn?.connectionId ?? '',
    owner?.profile ?? conn?.profile ?? '',
    sessionId ?? ''
  ])

  // `reused`: the URL came from an earlier mount, so its ticket can have died with a gateway restart
  // and its first error gets one fresh URL even before it loaded.
  const [media, setMedia] = useState(() => {
    const cached = cachedPlaybackSrc(cacheKey)

    return { failed: false, path, reused: Boolean(cached), src: cached ?? '' }
  })

  const loadedSrc = useRef('')
  // Minted for a source that failed before it ever loaded; it gets no retry of its own.
  const retriedSrc = useRef('')
  const resume = useRef<null | ResumePoint>(null)

  useEffect(() => {
    let cancelled = false
    let objectUrl = ''
    const cached = kind === 'file' ? undefined : cachedPlaybackSrc(cacheKey)

    setMedia(current =>
      cached && current.path === path && current.src === cached
        ? current
        : { failed: kind === 'file', path, reused: Boolean(cached), src: cached ?? '' }
    )

    if (kind === 'file' || cached) {
      return
    }

    void resolveMediaPlaybackSrc(path, owner, sessionId)
      .then(value => {
        if (value.startsWith('blob:')) {
          objectUrl = value
        } else {
          rememberPlaybackSrc(cacheKey, value)
        }

        if (!cancelled) {
          setMedia({ failed: false, path, reused: false, src: value })
        } else if (objectUrl) {
          URL.revokeObjectURL(objectUrl)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setMedia({ failed: true, path, reused: false, src: '' })
        }
      })

    return () => {
      cancelled = true

      if (objectUrl) {
        URL.revokeObjectURL(objectUrl)
      }
    }
  }, [cacheKey, kind, owner, path, sessionId])

  const onLoadedMetadata = useCallback((event: PlaybackEvent) => {
    const element = event.currentTarget
    const loaded = element.getAttribute('src') ?? ''
    const point = resume.current
    loadedSrc.current = loaded

    if (point?.src !== loaded) {
      return
    }

    resume.current = null
    element.currentTime = point.time

    if (!point.paused) {
      void element.play().catch(() => {
        // Resuming without a fresh gesture can be refused; the position is restored.
      })
    }
  }, [])

  const onError = useCallback(
    (event: PlaybackEvent) => {
      const element = event.currentTarget
      const failing = element.getAttribute('src') ?? ''

      const fail = () => {
        forgetPlaybackSrc(cacheKey)
        setMedia(current => (current.src === failing ? { ...current, failed: true } : current))
      }

      const neverLoaded = loadedSrc.current !== failing
      // A source that never loaded can only have expired if its element waited to fetch it
      // (preload="none" until played); it gets one fresh ticket. Blob sources never expire.
      const reused = media.reused && media.src === failing

      if (reused) {
        forgetPlaybackSrc(cacheKey)
      }

      const realFailure =
        neverLoaded && !reused && (element.preload !== 'none' || retriedSrc.current === failing)

      if (failing.startsWith('blob:') || realFailure) {
        fail()

        return
      }

      const point = { element, paused: element.paused, time: element.currentTime }

      void resolveMediaPlaybackSrc(path, owner, sessionId)
        .then(fresh => {
          if (fresh === failing || fresh.startsWith('blob:')) {
            if (fresh !== failing) {
              URL.revokeObjectURL(fresh)
            }

            fail()

            return
          }

          if (neverLoaded) {
            retriedSrc.current = fresh
          }

          rememberPlaybackSrc(cacheKey, fresh)
          resume.current = { ...point, src: fresh }
          setMedia(current => (current.src === failing ? { ...current, reused: false, src: fresh } : current))
        })
        .catch(fail)
    },
    [cacheKey, media.reused, media.src, owner, path, sessionId]
  )

  // An element with preload="none" fetches nothing for a new source until asked, so one
  // that was playing is asked again; onLoadedMetadata then restores the position.
  useEffect(() => {
    const point = resume.current

    if (!point || point.paused || point.src !== media.src || point.element.preload !== 'none') {
      return
    }

    void point.element.play().catch(() => {
      // Without the user's gesture still active the browser can refuse; Play works again.
    })
  }, [media.src])

  // Never paint the previous path's source while the new one resolves.
  const current = media.path === path

  return {
    failed: current ? media.failed : kind === 'file',
    kind,
    onError,
    onLoadedMetadata,
    src: current ? media.src : ''
  }
}
