import { useEffect, useRef } from 'react'

// Removing a playing media element alone can leave Chromium buffering its old
// source. Reset detached remote players so the protocol body is cancelled.
export function useMediaElementRef<T extends HTMLMediaElement>(src: string | undefined) {
  const ref = useRef<T>(null)

  useEffect(() => {
    const media = ref.current

    if (!media || !src?.startsWith('hermes-media://remote/')) {
      return
    }

    return () => {
      // React has already installed a replacement source before passive cleanup.
      // Reset the old player, but restore that new source; don't interrupt a
      // still-mounted unchanged player during StrictMode effect replay.
      const replacement = media.getAttribute('src')

      if (media.isConnected && replacement === src) {
        return
      }

      const playbackRate = media.playbackRate
      media.pause()
      media.removeAttribute('src')
      media.load()

      if (media.isConnected && replacement) {
        media.setAttribute('src', replacement)
        media.playbackRate = playbackRate
      }
    }
  }, [src])

  return ref
}
