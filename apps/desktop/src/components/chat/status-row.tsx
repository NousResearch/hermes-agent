import './status-stack.css'

import {
  type CSSProperties,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
  type ReactNode,
  type Ref,
  useRef
} from 'react'

import { cn } from '@/lib/utils'

import { StatusDismissButton } from './status-dismiss-button'

interface StatusRowProps {
  children: ReactNode
  className?: string
  depth?: number
  dismiss?: { label: string; onDismiss: () => void }
  expanded?: boolean
  /** Leading glyph slot (spinner / status dot / selection circle). */
  leading?: ReactNode
  /** Makes the whole row activatable (adds `cursor-pointer` + keyboard a11y).
   *  Receives the originating event so consumers can branch on modifier keys
   *  (e.g. ⌘/Ctrl-click). Trailing-slot buttons should `stopPropagation` so
   *  they don't also fire it. */
  onActivate?: (event: KeyboardEvent | MouseEvent) => void
  /** Right-aligned actions. Revealed on row hover/focus unless `trailingVisible`. */
  trailing?: ReactNode
  trailingVisible?: boolean
  /** Forwarded to the row's root — lets a wrapper (e.g. a context-menu trigger
   *  using `asChild`) attach `ref` / `onContextMenu` to the real DOM node. */
  ref?: Ref<HTMLDivElement>
  onContextMenu?: (event: MouseEvent) => void
}

/**
 * Shared row chrome for everything in the composer status stack — status items
 * (subagents, background) and queued prompts. Icons and dismiss controls align
 * with the first text line; section nesting and row padding share one CSS grid.
 */
export function StatusRow({
  children,
  className,
  depth = 0,
  dismiss,
  expanded,
  leading,
  onActivate,
  onContextMenu,
  ref,
  trailing,
  trailingVisible = false
}: StatusRowProps) {
  const gesture = useRef<{ x: number; y: number; dragged: boolean } | null>(null)

  const startGesture = (event: MouseEvent | PointerEvent) => {
    gesture.current = { x: event.clientX, y: event.clientY, dragged: false }
  }

  const moveGesture = (event: MouseEvent | PointerEvent) => {
    const start = gesture.current

    if (start && (event.clientX - start.x) ** 2 + (event.clientY - start.y) ** 2 > 16) {
      start.dragged = true
    }
  }

  return (
    <div
      aria-expanded={expanded}
      className={cn(
        'group/status-row status-row flex min-h-6 items-center gap-2 rounded-md px-1.5 py-1',
        // row-hover bundles cursor:pointer — only when the row actually activates.
        onActivate ? 'row-hover' : 'hover:bg-(--ui-row-hover-background)',
        className
      )}
      data-slot="status-row"
      data-status-icon={leading !== undefined ? '' : undefined}
      onClick={
        onActivate
          ? event => {
              const start = gesture.current
              gesture.current = null
              const selection = window.getSelection()

              // Only the click completing a text-selection drag is suppressed;
              // a later click can activate even if the browser keeps the selection.
              if (
                event.detail > 0 &&
                start &&
                (start.dragged || (event.clientX - start.x) ** 2 + (event.clientY - start.y) ** 2 > 16) &&
                selection &&
                !selection.isCollapsed &&
                selection.rangeCount > 0 &&
                selection.getRangeAt(0).intersectsNode(event.currentTarget)
              ) {
                return
              }

              onActivate(event)
            }
          : undefined
      }
      onContextMenu={onContextMenu}
      onKeyDown={
        onActivate
          ? event => {
              if (event.target === event.currentTarget && (event.key === 'Enter' || event.key === ' ')) {
                event.preventDefault()
                onActivate(event)
              }
            }
          : undefined
      }
      onMouseDown={onActivate ? startGesture : undefined}
      onMouseMove={onActivate ? moveGesture : undefined}
      onPointerDown={onActivate ? startGesture : undefined}
      onPointerMove={onActivate ? moveGesture : undefined}
      ref={ref}
      role={onActivate ? 'button' : undefined}
      style={{ '--status-row-depth': depth } as CSSProperties}
      tabIndex={onActivate ? 0 : undefined}
    >
      {dismiss && (
        <span className="status-row-dismiss">
          <StatusDismissButton label={dismiss.label} onDismiss={dismiss.onDismiss} />
        </span>
      )}
      {leading !== undefined && (
        <span className="status-row-icon flex size-3.5 shrink-0 items-center justify-center">{leading}</span>
      )}
      <div className="status-row-content flex min-w-0 flex-1 items-center gap-2">{children}</div>
      {trailing && (
        <div
          className={cn(
            'status-row-actions flex shrink-0 items-center gap-0.5',
            !trailingVisible && 'opacity-0 group-hover/status-row:opacity-100 group-focus-within/status-row:opacity-100'
          )}
        >
          {trailing}
        </div>
      )}
    </div>
  )
}
