import { useStore } from '@nanostores/react'
import {
  Children,
  type CSSProperties,
  isValidElement,
  type ReactElement,
  type ReactNode,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef
} from 'react'

import { cn } from '@/lib/utils'
import { $paneStates, ensurePaneRegistered, setPaneWidthOverride } from '@/store/panes'

import { PaneShellContext, type PaneShellContextValue, type PaneSlot } from './context'

type PaneSide = 'left' | 'right'
type WidthValue = string | number

interface PaneRoleMarker {
  __paneShellRole?: 'pane' | 'main'
}

export interface PaneProps {
  children?: ReactNode
  className?: string
  defaultOpen?: boolean
  /** Forces the pane closed (track→0, aria-hidden) without writing to the store — for transient route gates. */
  disabled?: boolean
  id: string
  maxWidth?: WidthValue
  minWidth?: WidthValue
  resizable?: boolean
  side: PaneSide
  width?: WidthValue
}

export interface PaneMainProps {
  children?: ReactNode
  className?: string
}

export interface PaneShellProps {
  children?: ReactNode
  className?: string
  style?: CSSProperties
}

interface CollectedPane {
  defaultOpen: boolean
  disabled: boolean
  id: string
  resizable: boolean
  side: PaneSide
  width: string
}

const DEFAULT_WIDTH = '16rem'
const DEFAULT_RESIZE_MIN_WIDTH = 160

const widthToCss = (value: WidthValue | undefined, fallback: string) =>
  value === undefined ? fallback : typeof value === 'number' ? `${value}px` : value

const remPx = () =>
  typeof window === 'undefined'
    ? 16
    : Number.parseFloat(window.getComputedStyle(document.documentElement).fontSize) || 16

// Resolves PaneProps.minWidth/maxWidth (number | "Npx" | "Nrem") to pixels for drag clamping.
function widthToPx(value: WidthValue | undefined) {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : undefined
  }

  const match = value?.trim().match(/^(-?\d*\.?\d+)(px|rem)?$/)

  if (!match) {
    return undefined
  }

  return Number.parseFloat(match[1]) * (match[2] === 'rem' ? remPx() : 1)
}

function isRole(child: unknown, role: 'pane' | 'main'): child is ReactElement {
  return isValidElement(child) && (child.type as PaneRoleMarker)?.__paneShellRole === role
}

function collectPanes(children: ReactNode) {
  const left: CollectedPane[] = []
  const right: CollectedPane[] = []
  let mainCount = 0

  Children.forEach(children, child => {
    if (isRole(child, 'main')) {
      mainCount++

      return
    }

    if (!isRole(child, 'pane')) {
      return
    }

    const props = child.props as PaneProps

    const entry: CollectedPane = {
      defaultOpen: props.defaultOpen ?? true,
      disabled: props.disabled ?? false,
      id: props.id,
      resizable: props.resizable ?? false,
      side: props.side,
      width: widthToCss(props.width, DEFAULT_WIDTH)
    }

    ;(props.side === 'left' ? left : right).push(entry)
  })

  return { left, mainCount, right }
}

function trackForPane(pane: CollectedPane, states: Record<string, { open: boolean; widthOverride?: number }>) {
  const stateOpen = states[pane.id]?.open ?? pane.defaultOpen
  const open = !pane.disabled && stateOpen

  if (!open) {
    return { open: false, track: '0px' }
  }

  const override = pane.resizable ? states[pane.id]?.widthOverride : undefined

  return { open: true, track: override !== undefined ? `${override}px` : pane.width }
}

export function PaneShell({ children, className, style }: PaneShellProps) {
  const paneStates = useStore($paneStates)
  const { left, mainCount, right } = useMemo(() => collectPanes(children), [children])

  if (import.meta.env.DEV && mainCount > 1) {
    console.warn('[PaneShell] expected at most one <PaneMain>, got', mainCount)
  }

  const ctxValue = useMemo(() => {
    const paneById = new Map<string, PaneSlot>()
    const tracks: string[] = []
    const cssVars: Record<string, string> = {}
    let column = 1

    for (const pane of left) {
      const { open, track } = trackForPane(pane, paneStates)
      tracks.push(track)
      paneById.set(pane.id, { column, open, side: 'left' })
      cssVars[`--pane-${pane.id}-width`] = track
      column++
    }

    tracks.push('minmax(0,1fr)')
    const mainColumn = column++

    for (const pane of right) {
      const { open, track } = trackForPane(pane, paneStates)
      tracks.push(track)
      paneById.set(pane.id, { column, open, side: 'right' })
      cssVars[`--pane-${pane.id}-width`] = track
      column++
    }

    return { cssVars, gridTemplate: tracks.join(' '), mainColumn, paneById } satisfies PaneShellContextValue & {
      cssVars: Record<string, string>
      gridTemplate: string
    }
  }, [left, paneStates, right])

  const composedStyle = useMemo<CSSProperties>(
    () => ({ ...ctxValue.cssVars, ...style, gridTemplateColumns: ctxValue.gridTemplate }),
    [ctxValue.cssVars, ctxValue.gridTemplate, style]
  )

  return (
    <PaneShellContext.Provider value={{ mainColumn: ctxValue.mainColumn, paneById: ctxValue.paneById }}>
      <div className={cn('relative grid h-full min-h-0', className)} style={composedStyle}>
        {children}
      </div>
    </PaneShellContext.Provider>
  )
}

export function Pane({
  children,
  className,
  defaultOpen = true,
  disabled = false,
  id,
  maxWidth,
  minWidth,
  resizable = false
}: PaneProps) {
  const ctx = useContext(PaneShellContext)
  const registered = useRef(false)
  const paneRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (registered.current) {
      return
    }

    registered.current = true
    ensurePaneRegistered(id, { open: defaultOpen })
  }, [defaultOpen, id])

  const slot = ctx?.paneById.get(id)
  const open = Boolean(slot?.open && !disabled)
  const canResize = open && resizable
  const lo = widthToPx(minWidth) ?? DEFAULT_RESIZE_MIN_WIDTH
  const hi = widthToPx(maxWidth) ?? Number.POSITIVE_INFINITY
  const side = slot?.side ?? 'left'

  const startResize = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const paneWidth = paneRef.current?.getBoundingClientRect().width ?? 0

      if (!canResize || paneWidth <= 0) {
        return
      }

      event.preventDefault()

      const handle = event.currentTarget
      const { pointerId, clientX: startX } = event
      const dir = side === 'left' ? 1 : -1
      const restoreCursor = document.body.style.cursor
      const restoreSelect = document.body.style.userSelect

      handle.setPointerCapture?.(pointerId)
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'

      const onMove = (e: PointerEvent) => {
        const next = paneWidth + (e.clientX - startX) * dir
        setPaneWidthOverride(id, Math.round(Math.min(hi, Math.max(lo, next))))
      }

      const cleanup = () => {
        document.body.style.cursor = restoreCursor
        document.body.style.userSelect = restoreSelect
        handle.releasePointerCapture?.(pointerId)
        window.removeEventListener('pointermove', onMove, true)
        window.removeEventListener('pointerup', cleanup, true)
        window.removeEventListener('pointercancel', cleanup, true)
        window.removeEventListener('blur', cleanup)
      }

      window.addEventListener('pointermove', onMove, true)
      window.addEventListener('pointerup', cleanup, true)
      window.addEventListener('pointercancel', cleanup, true)
      window.addEventListener('blur', cleanup)
    },
    [canResize, hi, id, lo, side]
  )

  if (!ctx) {
    if (import.meta.env.DEV) {
      console.warn(`[Pane:${id}] must be rendered inside <PaneShell>`)
    }

    return null
  }

  if (!slot) {
    return null
  }

  return (
    <div
      aria-hidden={!open}
      className={cn('relative row-start-1 min-w-0 overflow-hidden', !open && 'pointer-events-none', className)}
      data-pane-id={id}
      data-pane-open={open ? 'true' : 'false'}
      data-pane-side={slot.side}
      ref={paneRef}
      style={{ gridColumn: `${slot.column} / ${slot.column + 1}` }}
    >
      {canResize && (
        <div
          aria-label={`Resize ${id}`}
          aria-orientation="vertical"
          className={cn(
            'group absolute bottom-0 top-0 z-20 w-1 cursor-col-resize [-webkit-app-region:no-drag]',
            slot.side === 'left' ? 'right-0 translate-x-1/2' : 'left-0 -translate-x-1/2'
          )}
          onPointerDown={startResize}
          role="separator"
          tabIndex={0}
        >
          <span className="absolute inset-y-0 left-1/2 w-(--vscode-sash-hover-size,0.25rem) -translate-x-1/2 bg-(--ui-sash-hover-border) opacity-0 transition-opacity duration-100 group-hover:opacity-100 group-focus-visible:opacity-100" />
        </div>
      )}
      {children}
    </div>
  )
}

;(Pane as unknown as PaneRoleMarker).__paneShellRole = 'pane'

export function PaneMain({ children, className }: PaneMainProps) {
  const ctx = useContext(PaneShellContext)

  if (!ctx) {
    if (import.meta.env.DEV) {
      console.warn('[PaneMain] must be rendered inside <PaneShell>')
    }

    return null
  }

  return (
    <div
      className={cn('row-start-1 flex min-h-0 min-w-0 flex-col overflow-hidden', className)}
      data-pane-main="true"
      style={{ gridColumn: `${ctx.mainColumn} / ${ctx.mainColumn + 1}` }}
    >
      {children}
    </div>
  )
}

;(PaneMain as unknown as PaneRoleMarker).__paneShellRole = 'main'
