import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import { TranscriptVideo } from '../src/components/chat/transcript-video'
import { useMediaElementRef } from '../src/components/chat/use-media-element-ref'

function Audio({ src }: { src: string }) {
  const ref = useMediaElementRef<HTMLAudioElement>(src)

  return <audio controls muted preload="none" ref={ref} src={src} />
}

const root = createRoot(document.getElementById('root')!)
Object.assign(window, {
  mountPlayers: (suffix = '') => {
    flushSync(() =>
      root.render(
        <>
          {Array.from({ length: 8 }, (_, i) => {
            const src = `hermes-media://remote/audio-${i}${suffix}.wav`

            return i % 2 ? (
              <TranscriptVideo controls key={i} muted preload="none" src={src} />
            ) : (
              <Audio key={i} src={src} />
            )
          })}
        </>
      )
    )
  },
  removePlayers: () => flushSync(() => root.render(null)),
  startPlayers: async () => {
    // Start/stop sequentially, but leave every paused element mounted. This
    // exercises buffering left behind by actual Chromium players, not DOM events.
    // Deliberately stay below Chromium's six HTTP/1.1 connections. Six players
    // left paused can still occupy every slot; this probe tests reclamation,
    // not a separate media pool or unlimited simultaneous playback.
    for (const player of Array.from(document.querySelectorAll<HTMLMediaElement>('audio,video')).slice(0, 4)) {
      await Promise.race([
        player.play(),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error(`play timeout: ${player.src}, readyState=${player.readyState}`)), 5000)
        )
      ])
      player.pause()
    }
  }
})
