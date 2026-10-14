import { useCallback, useRef, useState } from 'react'

import type { ProbeResult } from './plugin-install-run'

const NO_PROBE: Promise<null> = Promise.resolve(null)

/** Inspect a repository once per request. `settled()` lets Install wait on an inspection still running, and a
 *  newer start (or `cancel`) turns every older answer stale, so a late result never paints over newer intent. */
export function usePluginProbe(unavailableMessage: string) {
  const [probe, setProbe] = useState<null | ProbeResult>(null)
  const token = useRef(0)
  const pending = useRef<Promise<null | ProbeResult>>(NO_PROBE)

  const start = useCallback(
    (repo: string) => {
      const mine = ++token.current
      const probeFn = window.hermesDesktop?.probePluginRepo

      const run: Promise<ProbeResult> = probeFn
        ? probeFn({ identifier: repo })
        : Promise.resolve({ ok: false, agent: false, desktop: false, warnings: [], error: unavailableMessage })

      setProbe(null)
      pending.current = run.then(result => {
        if (mine !== token.current) {
          return null
        }

        setProbe(result)

        return result
      })
    },
    [unavailableMessage]
  )

  const cancel = useCallback(() => {
    token.current += 1
    pending.current = NO_PROBE
    setProbe(null)
  }, [])

  return { probe, start, cancel, settled: () => pending.current }
}
