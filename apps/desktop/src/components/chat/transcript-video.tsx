import { useStore } from '@nanostores/react'
import { type ComponentProps, useCallback, useEffect, useRef } from 'react'

import { $videoPlaybackSpeed, setVideoPlaybackSpeed } from '@/store/video-playback-speed'

// A transcript <video> that remembers the playback rate. The native controls'
// rate menu is the only speed UI; picking a rate there persists it as the
// device-level preference every later player (and other open windows) starts
// from. Ported from block/buzz#7336.
export function TranscriptVideo(props: ComponentProps<'video'>) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const speed = useStore($videoPlaybackSpeed)

  useEffect(() => {
    const video = videoRef.current

    if (video) {
      video.playbackRate = speed
    }
  }, [speed])

  // ratechange also fires when WE set the rate (mount, cross-window sync), so
  // only a rate that differs from the preference — i.e. one the user picked in
  // the controls — persists. setVideoPlaybackSpeed drops out-of-range values.
  const onRateChange = useCallback(() => {
    const video = videoRef.current

    if (video && video.playbackRate !== $videoPlaybackSpeed.get()) {
      setVideoPlaybackSpeed(video.playbackRate)
    }
  }, [])

  return <video onRateChange={onRateChange} ref={videoRef} {...props} />
}
