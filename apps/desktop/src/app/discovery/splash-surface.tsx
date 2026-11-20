import './discovery.css'

import { type ReactNode, type RefObject, useCallback, useEffect, useId, useRef, useState } from 'react'

import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from '@/lib/escape-layers'
import { useRootAttribute } from '@/lib/root-attribute'

import { type NavSpot, NavSpotlight } from './nav-spotlight'

export interface SplashRow<Id extends string = string> {
  id: Id
  /** The row's glyph: the place's own. */
  glyph: ReactNode
  /** The row's name line. */
  label: ReactNode
  /** The row's accessible name: what `label` and `meta` say together, in words. */
  name: string
  /** The row's second line, under its name. */
  line: string
  /** Trailing detail on the row (a count). */
  meta?: ReactNode
  /** The place in the nav the row lights up while active. */
  spot?: NavSpot
}

export interface SplashSurfaceProps<Id extends string> {
  /** Which splash this is, for its own styling (`data-splash`). */
  kind: string
  eyebrow: string
  title: string
  subtitle: string
  /** The rows' list, for assistive tech. */
  listLabel: string
  rows: readonly SplashRow<Id>[]
  /** Under the rows: the splash's own way out (a button, a sign-up). */
  actions: ReactNode
  /** Under the actions: how else it closes. */
  footnote: string
  /** Escape was pressed while nothing above the splash wanted it. */
  onEscape: () => void
  /** A row became active by the person's hand (hover, focus or click), not by default. */
  onRowActivated?: (id: Id) => void
  /** Take focus on mount: when the person opened it, or when nothing else has focus. */
  takeFocus: boolean
}

/**
 * What every splash does as a whole: it takes focus when the person opened it (or when nothing else
 * has focus, or focus sat in the chat it just made inert), so Tab starts inside it; and Escape closes
 * it when nothing above it wants the key. Returns the ref for the splash's own element.
 */
export function useSplashFrame(takeFocus: boolean, onEscape: () => void): RefObject<HTMLElement | null> {
  const surfaceRef = useRef<HTMLElement>(null)

  // While a splash is up the rail's own selection stands down (discovery.css).
  useRootAttribute('data-discovery-splash', true)

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

  return surfaceRef
}

/**
 * A splash on the stage in the welcome's language (Figma E5 18677:29814): the 640 column with a
 * heading, one card of rows (glyph, name, line, count), the way out and a footnote. A row a pointer,
 * focus or tap reaches lights its place in the nav; nothing is lit until then, and a tap leaves it
 * on that row (touch has no hover). The rename splash is one.
 */
export function SplashSurface<Id extends string>({
  actions,
  eyebrow,
  footnote,
  kind,
  listLabel,
  onEscape,
  onRowActivated,
  rows,
  subtitle,
  takeFocus,
  title
}: SplashSurfaceProps<Id>) {
  const id = useId()
  const surfaceRef = useSplashFrame(takeFocus, onEscape)
  const activeRowRef = useRef<HTMLElement | null>(null)

  // The active row's button, for the spotlight's connector.
  const bindActiveRow = useCallback((node: HTMLElement | null) => {
    activeRowRef.current = node
  }, [])

  const [picked, setPicked] = useState<Id | null>(null)
  const [hovered, setHovered] = useState<Id | null>(null)
  const [focused, setFocused] = useState<Id | null>(null)
  const activeId = hovered ?? focused ?? picked
  const active = rows.find(row => row.id === activeId)

  const activate = (rowId: Id, set: (rowId: Id) => void) => {
    set(rowId)
    onRowActivated?.(rowId)
  }

  return (
    <section
      aria-describedby={`${id}-subtitle`}
      aria-labelledby={`${id}-title`}
      className="discovery-splash"
      data-slot="discovery-splash"
      data-splash={kind}
      ref={surfaceRef}
      tabIndex={-1}
    >
      <div className="discovery-welcome">
        <p className="discovery-welcome-eyebrow">{eyebrow}</p>
        <div className="discovery-welcome-heading">
          <h1 className="discovery-welcome-title" id={`${id}-title`}>
            {title}
          </h1>
          <p className="discovery-welcome-subtitle" data-tone="regular" id={`${id}-subtitle`}>
            {subtitle}
          </p>
        </div>
        <ul
          aria-label={listLabel}
          className="discovery-welcome-card discovery-welcome-places"
          onPointerLeave={() => setHovered(null)}
        >
          {rows.map(row => {
            const isActive = row.id === active?.id

            return (
              <li data-active={isActive || undefined} key={row.id}>
                <button
                  aria-label={`${row.name}: ${row.line}`}
                  className="discovery-welcome-place"
                  onBlur={() => setFocused(current => (current === row.id ? null : current))}
                  onClick={() => activate(row.id, setPicked)}
                  onFocus={() => activate(row.id, setFocused)}
                  onPointerEnter={event => {
                    if (event.pointerType === 'mouse') {
                      activate(row.id, setHovered)
                    }
                  }}
                  ref={isActive ? bindActiveRow : undefined}
                  type="button"
                >
                  {row.glyph}
                  <span aria-hidden className="discovery-welcome-place-text">
                    <span className="discovery-welcome-place-name">{row.label}</span>
                    <span className="discovery-welcome-place-line">{row.line}</span>
                  </span>
                  {row.meta !== undefined && (
                    <span aria-hidden className="discovery-welcome-place-meta">
                      {row.meta}
                    </span>
                  )}
                </button>
              </li>
            )
          })}
        </ul>
        <div className="discovery-welcome-actions">{actions}</div>
        <p className="discovery-welcome-footnote">{footnote}</p>
      </div>
      {active?.spot && <NavSpotlight anchor={activeRowRef} key={active.id} spot={active.spot} />}
    </section>
  )
}
