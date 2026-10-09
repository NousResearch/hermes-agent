import { type SyntheticEvent, useCallback, useEffect, useRef, useState } from 'react'

import { mediaKind, resolveMediaPlaybackSrc } from '@/lib/media'

type PlaybackEvent = SyntheticEvent<HTMLAudioElement | HTMLVideoElement>

interface ResumePoint {
  paused: boolean
  src: string
  time: number
}

/** The playable source for a transcript audio/video attachment.
 *
 * Webapp sources are stream-ticket URLs that expire, and the element needs a
 * valid one for every later range request (a seek, or play after a long
 * pause). So a source that has loaded is re-resolved on its next error and
 * playback resumes where it stopped. A source that never loaded, or that
 * resolves to the same URL again, is a real failure. */
export function useMediaPlaybackSrc(path: string) {
  const kind = mediaKind(path)
  const [media, setMedia] = useState({ failed: false, path, src: '' })
  const loadedSrc = useRef('')
  const resume = useRef<null | ResumePoint>(null)

  useEffect(() => {
    let cancelled = false
    let objectUrl = ''

    setMedia({ failed: kind === 'file', path, src: '' })

    if (kind === 'file') {
      return
    }

    void resolveMediaPlaybackSrc(path)
      .then(value => {
        if (value.startsWith('blob:')) {
          objectUrl = value
        }

        if (!cancelled) {
          setMedia({ failed: false, path, src: value })
        } else if (objectUrl) {
          URL.revokeObjectURL(objectUrl)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setMedia({ failed: true, path, src: '' })
        }
      })

    return () => {
      cancelled = true

      if (objectUrl) {
        URL.revokeObjectURL(objectUrl)
      }
    }
  }, [kind, path])

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

      const fail = () =>
        setMedia(current => (current.src === failing ? { ...current, failed: true } : current))

      // Only a source that has loaded can have expired; blob sources never do.
      if (loadedSrc.current !== failing || failing.startsWith('blob:')) {
        fail()

        return
      }

      const point = { paused: element.paused, time: element.currentTime }

      void resolveMediaPlaybackSrc(path)
        .then(fresh => {
          if (fresh === failing || fresh.startsWith('blob:')) {
            if (fresh !== failing) {
              URL.revokeObjectURL(fresh)
            }

            fail()

            return
          }

          resume.current = { ...point, src: fresh }
          setMedia(current => (current.src === failing ? { ...current, src: fresh } : current))
        })
        .catch(fail)
    },
    [path]
  )

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
