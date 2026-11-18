import './discovery.css'

import { type ReactNode, useEffect, useId, useRef, useState } from 'react'

import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from '@/lib/escape-layers'

import { type NavSpot, NavSpotlight } from './nav-spotlight'

export interface SplashRow<Id extends string = string> {
  id: Id
  /** The row's one visible line. */
  label: ReactNode
  /** The row's accessible name: what `label` and `meta` say together, in words. */
  name: string
  /** Trailing detail on the row's line (a count). */
  meta?: ReactNode
  /** The sentence shown while the row is hovered, focused or picked. */
  detail?: string
  /** The place in the nav the row lights up while active. */
  spot?: NavSpot
}

export interface SplashSurfaceProps<Id extends string> {
  eyebrow: string
  title: string
  subtitle: string
  /** The rows' list, for assistive tech. */
  listLabel: string
  rows: readonly SplashRow<Id>[]
  /** Under the rows: the splash's own way out (a button, a sign-up). */
  footer: ReactNode
  /** Escape was pressed while nothing above the splash wanted it. */
  onEscape: () => void
  /** A row became active by the person's hand (hover, focus or click), not by default. */
  onRowActivated?: (id: Id) => void
  /** Take focus on mount: when the person opened it, or when nothing else has focus. */
  takeFocus: boolean
}

/**
 * A splash that fills the chat area while the nav stays live: a heading, a list of rows that each
 * say one thing (and say more, and light up their place in the nav, while active), and a footer.
 * The rename splash is one; the first-time splash is another. The first row starts active, so the
 * splash shows what it does before anyone touches it; a pointer or focus moves the highlight, a
 * click or tap leaves it on that row (touch has no hover).
 */
export function SplashSurface<Id extends string>({
  eyebrow,
  footer,
  listLabel,
  onEscape,
  onRowActivated,
  rows,
  subtitle,
  takeFocus,
  title
}: SplashSurfaceProps<Id>) {
  const id = useId()
  const surfaceRef = useRef<HTMLElement>(null)
  const activeRowRef = useRef<HTMLButtonElement>(null)
  const [picked, setPicked] = useState<Id | null>(rows[0]?.id ?? null)
  const [hovered, setHovered] = useState<Id | null>(null)
  const [focused, setFocused] = useState<Id | null>(null)
  const activeId = hovered ?? focused ?? picked
  const active = rows.find(row => row.id === activeId)

  const activate = (rowId: Id, set: (rowId: Id) => void) => {
    set(rowId)
    onRowActivated?.(rowId)
  }

  useEffect(() => {
    const focus = document.activeElement
    // Focus left where it was unless it was nowhere, or in the chat the splash just made inert.
    const stranded = !focus || focus === document.body || Boolean(focus.closest('[inert]'))

    if (!takeFocus && !stranded) {
      return
    }

    // A frame later, outside React's commit: the tray that reopened the splash is still closing, and
    // moving focus out of a Radix layer while React renders makes it flush synchronously.
    const frame = requestAnimationFrame(() => surfaceRef.current?.focus({ preventScroll: true }))

    return () => cancelAnimationFrame(frame)
  }, [takeFocus])

  useEffect(() => {
    const surface = surfaceRef.current
    const release = pushEscapeLayer(ESCAPE_PRIORITY.splash)

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !event.defaultPrevented && isTopEscapeLayer(ESCAPE_PRIORITY.splash)) {
        event.preventDefault()
        onEscape()
      }
    }

    // From inside the splash it is the splash's Escape before the panes around it see the key (a
    // compact layout's pane handles Escape too); from the nav or nowhere, the window's.
    surface?.addEventListener('keydown', onKeyDown)
    window.addEventListener('keydown', onKeyDown)

    return () => {
      surface?.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keydown', onKeyDown)
      release()
    }
  }, [onEscape])

  return (
    <section
      aria-describedby={`${id}-subtitle`}
      aria-labelledby={`${id}-title`}
      className="discovery-splash"
      data-slot="discovery-splash"
      ref={surfaceRef}
      tabIndex={-1}
    >
      <span aria-hidden className="discovery-splash-brackets" />
      <div className="discovery-splash-content">
        <p className="discovery-splash-eyebrow">{eyebrow}</p>
        <h1 className="discovery-splash-title" id={`${id}-title`}>
          {title}
        </h1>
        <p className="discovery-splash-subtitle" id={`${id}-subtitle`}>
          {subtitle}
        </p>
        <ul aria-label={listLabel} className="discovery-splash-rows" onPointerLeave={() => setHovered(null)}>
          {rows.map(row => {
            const isActive = row.id === active?.id
            const detailId = `${id}-${row.id}-detail`

            return (
              <li className="discovery-splash-row" data-active={isActive || undefined} key={row.id}>
                <button
                  aria-controls={row.detail ? detailId : undefined}
                  aria-expanded={row.detail ? isActive : undefined}
                  aria-label={row.name}
                  className="discovery-splash-row-button"
                  onBlur={() => setFocused(current => (current === row.id ? null : current))}
                  onClick={() => activate(row.id, setPicked)}
                  onFocus={() => activate(row.id, setFocused)}
                  onPointerEnter={event => {
                    if (event.pointerType === 'mouse') {
                      activate(row.id, setHovered)
                    }
                  }}
                  ref={isActive ? activeRowRef : undefined}
                  type="button"
                >
                  <span className="discovery-splash-row-label">{row.label}</span>
                  {row.meta !== undefined && (
                    <span aria-hidden className="discovery-splash-row-meta">
                      {row.meta}
                    </span>
                  )}
                </button>
                {row.detail && (
                  <p className="discovery-splash-row-detail" hidden={!isActive} id={detailId}>
                    {row.detail}
                  </p>
                )}
              </li>
            )
          })}
        </ul>
        <div className="discovery-splash-footer">{footer}</div>
      </div>
      {active?.spot && <NavSpotlight anchor={activeRowRef} key={active.id} spot={active.spot} />}
    </section>
  )
}
