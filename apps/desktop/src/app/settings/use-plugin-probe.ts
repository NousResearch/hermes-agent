import { useCallback, useEffect, useRef, useState } from 'react'

import type { ProbeResult } from './plugin-install-run'

const NO_PROBE: Promise<null> = Promise.resolve(null)

export type GatewayProbe = (repo: string) => Promise<ProbeResult>

/** Inspect a repository once per request. `settled()` lets Install wait on an inspection still running, and a
 *  newer start (or `cancel`) turns every older answer stale, so a late result never paints over newer intent.
 *  Native Desktop inspects in Electron; a renderer without that door (the browser-hosted app) asks the
 *  backend through `gatewayProbe`. With neither, inspection is `unavailable`. */
export function usePluginProbe(unavailableMessage: string, gatewayProbe?: GatewayProbe) {
  const [probe, setProbe] = useState<null | ProbeResult>(null)
  const token = useRef(0)
  const pending = useRef<Promise<null | ProbeResult>>(NO_PROBE)
  // The newest door, read when an inspection starts: a fresh closure each render must not restart one.
  const gateway = useRef(gatewayProbe)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    gateway.current = gatewayProbe
  }, [gatewayProbe])

  const start = useCallback(
    (repo: string) => {
      const mine = ++token.current
      const probeFn = window.hermesDesktop?.probePluginRepo
      const viaGateway = gateway.current

      const run: Promise<ProbeResult> = probeFn
        ? probeFn({ identifier: repo })
        : viaGateway
          ? viaGateway(repo)
          : Promise.resolve({
              ok: false,
              agent: false,
              desktop: false,
              warnings: [],
              unavailable: true,
              error: unavailableMessage
            })

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
