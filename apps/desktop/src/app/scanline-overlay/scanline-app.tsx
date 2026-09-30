import './scanline.css'

import { useEffect, useState } from 'react'

import type { ScanlineState } from '@/lib/scanline'

export function ScanlineApp() {
  const [state, setState] = useState<ScanlineState>('hidden')

  useEffect(() => {
    const api = window.hermesDesktop?.scanline
    let mounted = true
    let receivedLiveState = false

    const unsubscribe = api?.onState(next => {
      if (mounted) {
        receivedLiveState = true
        setState(next)
      }
    })

    void api
      ?.getState()
      .then(next => {
        if (mounted && !receivedLiveState) {
          setState(next)
        }
      })
      .catch(() => undefined)

    return () => {
      mounted = false
      unsubscribe?.()
    }
  }, [])

  return (
    <main aria-hidden className="scanline-surface" data-state={state}>
      <div className="scanline-border" />
      <div className="scanline-sweeper">
        <div className="scanline-trail" />
        <div className="scanline-core" />
      </div>
    </main>
  )
}
