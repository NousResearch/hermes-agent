import { useStore } from '@nanostores/react'
import { type ComponentProps, useCallback, useEffect } from 'react'

import { $videoPlaybackSpeed, setVideoPlaybackSpeed } from '@/store/video-playback-speed'

import { useMediaElementRef } from './use-media-element-ref'

// Playback position per source, remembered across remounts. Transcript rows
// legitimately unmount while a turn streams (the render-budget slice recycles
// older rows; the streaming markdown re-parse re-mounts AST leaves), and a
// fresh <video> restarted at 0 read as the clip "reloading mid-playback"
// (#123018). Process-memory only: the map deliberately holds a bounded number
// of recent sources and is dropped on reload.
const MAX_REMEMBERED_SOURCES = 24
const rememberedPositions = new Map<string, { paused: boolean; time: number }>()

export function rememberVideoPosition(src: string, time: number, paused: boolean): void {
  if (!src || !Number.isFinite(time) || time <= 0) {
    return
  }

  rememberedPositions.delete(src)
  rememberedPositions.set(src, { paused, time })

  if (rememberedPositions.size > MAX_REMEMBERED_SOURCES) {
    const oldest = rememberedPositions.keys().next().value

    if (oldest !== undefined) {
      rememberedPositions.delete(oldest)
    }
  }
}

export function recallVideoPosition(src: string): { paused: boolean; time: number } | undefined {
  return rememberedPositions.get(src)
}

// A transcript <video> that remembers the playback rate. The native controls'
// rate menu is the only speed UI; picking a rate there persists it as the
// device-level preference every later player (and other open windows) starts
// from. Ported from block/buzz#7336.
export function TranscriptVideo(props: ComponentProps<'video'>) {
  const videoRef = useMediaElementRef<HTMLVideoElement>(props.src)
  const speed = useStore($videoPlaybackSpeed)
  const src = typeof props.src === 'string' ? props.src : ''

  useEffect(() => {
    const video = videoRef.current

    if (video) {
      video.playbackRate = speed
    }
  }, [speed, videoRef])

  // Restore the last position for this source when the element (re)mounts —
  // a remount mid-turn otherwise restarts the clip from the top.
  useEffect(() => {
    const video = videoRef.current
    const remembered = src ? recallVideoPosition(src) : undefined

    if (video && remembered) {
      video.currentTime = remembered.time

      if (!remembered.paused) {
        void video.play().catch(() => {
          // Autoplay without a gesture can be refused (muted policy); the
          // user can press play — the position is already restored.
        })
      }
    }
  }, [src])

  // Keep the remembered position fresh while the element lives.
  useEffect(() => {
    const video = videoRef.current

    if (!video || !src) {
      return
    }

    const commit = () => {
      rememberVideoPosition(src, video.currentTime, video.paused)
    }

    video.addEventListener('timeupdate', commit)
    video.addEventListener('pause', commit)
    video.addEventListener('ended', commit)

    return () => {
      commit()

      video.removeEventListener('timeupdate', commit)
      video.removeEventListener('pause', commit)
      video.removeEventListener('ended', commit)
    }
  }, [src])

  // ratechange also fires when WE set the rate (mount, cross-window sync), so
  // only a rate that differs from the preference — i.e. one the user picked in
  // the controls — persists. setVideoPlaybackSpeed drops out-of-range values.
  const onRateChange = useCallback(() => {
    const video = videoRef.current

    if (video && video.playbackRate !== $videoPlaybackSpeed.get()) {
      setVideoPlaybackSpeed(video.playbackRate)
    }
  }, [videoRef])

  return <video onRateChange={onRateChange} ref={videoRef} {...props} />
}
