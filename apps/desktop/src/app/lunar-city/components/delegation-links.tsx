import { useCallback, useEffect, useRef, useState } from 'react'

import type { LunarCitySnapshot, LunarEntity } from '../model'
import { createWorkerDelegationController, type DelegationEnvironment, type DelegationLink } from '../worker-delegation'

export interface DelegationLinksProps extends DelegationEnvironment {
  snapshot: LunarCitySnapshot
  visible: boolean
  enabled: boolean
  reducedMotion: boolean
  onSelectEntity(entity: LunarEntity): void
}

export function DelegationLinks({
  snapshot,
  visible,
  enabled,
  reducedMotion,
  presentation,
  project,
  onSelectEntity
}: DelegationLinksProps) {
  const controller = useRef(createWorkerDelegationController())
  const [links, setLinks] = useState<readonly DelegationLink[]>([])
  const published = useRef<readonly DelegationLink[]>([])

  const publishLinks = useCallback((next: readonly DelegationLink[]) => {
    const current = published.current

    const unchanged = current.length === next.length && current.every((link, index) => {
      const other = next[index]!

      return link.parentKey === other.parentKey && link.childKey === other.childKey && link.kind === other.kind &&
        link.from.x === other.from.x && link.from.y === other.from.y && link.to.x === other.to.x && link.to.y === other.to.y
    })

    if (!unchanged) {
      published.current = next
      setLinks(next)
    }
  }, [])

  useEffect(() => {
    if (!visible || !enabled || reducedMotion) {
      controller.current.clear()
      publishLinks([])

      return
    }

    controller.current.update(snapshot)

    if (!controller.current.hasCandidates()) {
      publishLinks([])

      return
    }

    let previous = performance.now()
    let emptyMs = 0

    // Let the source listener apply this snapshot and navigation start before sampling.
    // A bounded grace period also covers asynchronous worker creation and route queries.
    const timer = setInterval(() => {
      const now = performance.now()
      const elapsed = now - previous
      previous = now
      const next = controller.current.links(elapsed, { presentation, project })
      emptyMs = next.length ? 0 : emptyMs + elapsed
      publishLinks(next)

      if (!controller.current.hasCandidates() || emptyMs >= 3000) {
        clearInterval(timer)
      }
    }, 100)

    return () => clearInterval(timer)
  }, [snapshot, visible, enabled, reducedMotion, presentation, project, publishLinks])

  const shown =
    visible && enabled && !reducedMotion
      ? links.filter(
          link =>
            snapshot.entities.get(link.parentKey)?.authority === 'authoritative' &&
            snapshot.entities.get(link.childKey)?.authority === 'authoritative'
        )
      : []

  return (
    <>
      <svg
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 h-full w-full"
        preserveAspectRatio="none"
        style={{ overflow: 'hidden' }}
        viewBox="0 0 100 100"
      >
        {shown.map(link => (
          <path
            d={`M ${link.from.x} ${link.from.y} L ${link.to.x} ${link.to.y}`}
            fill="none"
            key={link.childKey}
            stroke={link.kind === 'return' ? '#a6e3b2' : '#a6c9ed'}
            strokeDasharray={link.kind === 'traveling' ? '5 5' : undefined}
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
      {shown.length ? (
        <section aria-label="Observed delegation links" className="lunar-city-delegation-legend">
          <p>Observed delegation · local visual links. No messages or artifacts are sent.</p>
          <ul>
            {shown.map(link => {
              const parent = snapshot.entities.get(link.parentKey)!,
                child = snapshot.entities.get(link.childKey)!

              const childId = child.identity.kind === 'subagent' ? child.identity.subagentId : child.key

              return (
                <li key={link.childKey}>
                  {link.kind === 'return'
                    ? 'New completion observed'
                    : link.kind === 'handoff'
                      ? 'Handoff state observed'
                      : 'Traveling toward the same destination'}
                  :{' '}
                  <button onClick={() => onSelectEntity(parent)} type="button">
                    Inspect parent · {parent.identity.profile} · {parent.identity.connectionId}
                  </button>{' '}
                  to{' '}
                  <button onClick={() => onSelectEntity(child)} type="button">
                    Inspect subagent {childId} · {child.identity.profile} · {child.identity.connectionId}
                  </button>
                </li>
              )
            })}
          </ul>
        </section>
      ) : null}
    </>
  )
}
