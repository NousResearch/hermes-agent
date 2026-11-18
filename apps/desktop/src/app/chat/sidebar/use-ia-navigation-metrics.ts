import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { $apiRequestScope } from '@/api/client'
import { $desktopMetricsGate } from '@/store/desktop-metrics'
import { captureIaRecorder } from '@/store/ia-metrics'
import { IA_AREAS, type IaArea } from '@/store/ia-metrics-contract'

import type { AppView } from '../../routes'

type Area = IaArea

export function iaArea(view: AppView): Area {
  if (view === 'extension') {
    return 'plugin'
  }

  return IA_AREAS.includes(view as Area) ? (view as Area) : 'other'
}

/** A visible visit; no raw route, session identity, or background-window dwell. */
export function useIaNavigationMetrics(view: AppView): void {
  const gate = useStore($desktopMetricsGate)
  const scope = useStore($apiRequestScope)
  const area = iaArea(view)
  const last = useRef<{ area: Area; from?: Area; at: number } | null>(null)
  // Visit/query history, not a mirror of current reactive state.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    last.current = null
  }, [gate, scope.connectionId, scope.profile])
  // Visit/query history, not a mirror of current reactive state.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    if (gate !== 'on') {
      return
    }

    const record = captureIaRecorder()
    let started: number | null = null

    const show = () => {
      if (document.visibilityState === 'hidden' || started !== null) {
        return
      }

      started = Date.now()
      const previous = last.current
      record(`ia.view.${area}`)

      if (previous && previous.area !== area) {
        // Both values are closed Area codes; the recorder additionally allowlists the pair.
        record(`ia.transition.${previous.area}_to_${area}`)

        if (previous.from === area && started - previous.at < 5000) {
          record(`ia.backtrack.${area}`)
        }
      }

      last.current = { area, from: previous?.area, at: started }
    }

    const hide = () => {
      if (started === null) {
        return
      }

      const elapsed = Date.now() - started
      const bucket = elapsed < 5000 ? 'lt5s' : elapsed < 30_000 ? '5s_to30s' : elapsed < 300_000 ? '30s_to5m' : 'gte5m'
      record(`ia.dwell.${area}_${bucket}`)
      started = null
    }

    const visibility = () => {
      if (document.visibilityState === 'hidden') {
        hide()
        last.current = null
      } else {
        show()
      }
    }

    show()
    document.addEventListener('visibilitychange', visibility)

    return () => {
      hide()
      document.removeEventListener('visibilitychange', visibility)
    }
  }, [area, gate, scope.connectionId, scope.profile])
}
