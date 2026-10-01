// Space → run / pause / resume. K does the same. Cmd/Ctrl+Shift+L tidies.
// Camera is scroll/pinch + the tidy/fit button — Space is a verb.
// (Undo/redo keys live in useUndoRedo.)

import { useCallback, useEffect } from 'react'

import type { usePlayer } from './player'

interface CanvasKeys {
  player: ReturnType<typeof usePlayer>
  tidy: () => void
}

export function useCanvasKeys({ player, tidy }: CanvasKeys) {
  const transport = useCallback(() => {
    // Parked on a person. Nothing the transport can do will move the run —
    // only the answer will — so the key that means "carry on" puts the
    // question back in front of you rather than doing nothing.
    if (player.asking) {
      player.reveal()

      return
    }

    if (!player.running) {
      player.start()
    } else if (player.pauseState === 'none') {
      player.requestPause()
    } else if (player.pauseState === 'paused') {
      player.resume()
    }
    // "pausing": the request is already in flight — the key waits with you.
  }, [player])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === 'l') {
        e.preventDefault()
        tidy()

        return
      }

      if (e.metaKey || e.ctrlKey || e.altKey) {
        return
      }

      const t = e.target as HTMLElement | null

      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) {
        return
      }

      if (e.code === 'Space' || e.key.toLowerCase() === 'k') {
        e.preventDefault()
        transport()
      }
    }

    window.addEventListener('keydown', onKey)

    return () => window.removeEventListener('keydown', onKey)
  }, [tidy, transport])
}
