import { useStore } from '@nanostores/react'
import {
  createContext,
  type CSSProperties,
  memo,
  type MouseEventHandler,
  type ReactNode,
  useCallback,
  useContext,
  useId,
  useLayoutEffect,
  useRef,
  useState
} from 'react'

import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import type { Contribution } from '@/contrib/types'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { cn } from '@/lib/utils'

import { $layoutEditMode } from '../../edit-mode'
import { hiddenPaneProps, PaneGroupContext, PaneLifecycleContext, PaneVisibleContext } from '../../pane-visibility'
import { $collapsedTreeSides, $treePaneEpochs, paneRootSide } from '../store'

import { paneChrome } from './track-model'

interface Placement {
  anchor: string
  groupId: string
  headerVisible: boolean
  visible: boolean
  overlay?: boolean
  onMouseLeave?: MouseEventHandler<HTMLDivElement>
}

type Placements = ReadonlyMap<string, Placement>
type PlacePane = (paneId: string, placement: Placement) => () => void
const PlacementContext = createContext<PlacePane | null>(null)

/** Own the live bodies outside the replaceable split/group tree. Neither React
 * remounts nor DOM moves are safe for a native guest: even moveBefore destroys
 * an Electron webview's guest. Only its CSS anchor and contexts may change. */
export function KeepAlivePanes({ children }: { children: ReactNode }) {
  const panes = useContributions('panes')
  const epochs = useStore($treePaneEpochs)
  const [placements, setPlacements] = useState<Placements>(() => new Map())

  const place = useCallback<PlacePane>((id, placement) => {
    setPlacements(previous => {
      // Restored background tabs are lazy until first activation.
      if (!placement.visible && !previous.has(id)) {
        return previous
      }

      return new Map(previous).set(id, placement)
    })

    return () => {
      setPlacements(previous => {
        // A retiring slot must not hide a newer placement of the same pane.
        if (previous.get(id) !== placement) {
          return previous
        }

        return new Map(previous).set(id, { ...placement, visible: false })
      })
    }
  }, [])

  useLayoutEffect(() => {
    setPlacements(previous => {
      // Contribution removal is destruction; placement removal is only Hide.
      const present = new Set(
        panes.filter(pane => pane.render && paneChrome(pane).lifecycleKeepAlive).map(pane => pane.id)
      )

      const retained = [...previous].filter(([id]) => present.has(id))

      return retained.length === previous.size ? previous : new Map(retained)
    })
  }, [panes])

  return (
    <PlacementContext value={place}>
      {children}
      {/* Map insertion order never follows tree order: React must not move a
          surviving guest host when panes or groups are rearranged. */}
      {[...placements].map(([id, placement]) => {
        const pane = panes.find(candidate => candidate.id === id)

        if (!pane?.render || !paneChrome(pane).lifecycleKeepAlive) {
          return null
        }

        return <KeepAlivePaneHost epoch={epochs[id] ?? 0} key={id} pane={pane} placement={placement} />
      })}
    </PlacementContext>
  )
}

interface KeepAlivePaneHostProps {
  pane: Contribution
  placement: Placement
  epoch: number
}

const KeepAlivePaneHost = memo(function KeepAlivePaneHost({ pane, placement, epoch }: KeepAlivePaneHostProps) {
  const ref = useRef<HTMLDivElement>(null)

  // Retain native guest coordinates while its placement is absent/minimized.
  // CSS follows the slot during resize; recording dimensions never re-renders
  // a guest or moves its connected host.
  const rememberSize = useCallback(
    (entries: readonly ResizeObserverEntry[]) => {
      const rect = entries[0]?.contentRect

      if (placement.visible && rect && rect.width > 0 && rect.height > 0 && ref.current) {
        ref.current.style.setProperty('--pane-kept-width', `${rect.width}px`)
        ref.current.style.setProperty('--pane-kept-height', `${rect.height}px`)
      }
    },
    [placement.visible]
  )

  useResizeObserver(rememberSize, ref)

  return (
    <div
      {...hiddenPaneProps(!placement.visible)}
      aria-hidden={!placement.visible || undefined}
      // Local stacking contains guest chrome below tree sashes/edit/drop veils.
      className={cn(
        'absolute overflow-auto',
        placement.overlay ? 'z-40' : 'z-0',
        !placement.visible && 'pointer-events-none invisible'
      )}
      data-pane-host={pane.id}
      data-pane-overlay={placement.overlay || undefined}
      data-tree-group={placement.groupId}
      data-zone-header={placement.headerVisible || undefined}
      inert={!placement.visible || undefined}
      onMouseLeave={placement.onMouseLeave}
      ref={ref}
      style={
        {
          positionAnchor: placement.anchor,
          left: 'anchor(left)',
          top: 'anchor(top)',
          width: placement.visible ? 'anchor-size(width)' : 'var(--pane-kept-width, 0px)',
          height: placement.visible ? 'anchor-size(height)' : 'var(--pane-kept-height, 0px)'
        } as CSSProperties
      }
    >
      <PaneGroupContext value={placement.groupId}>
        <PaneLifecycleContext value={placement.visible ? 'visible' : 'hot-hidden'}>
          <PaneVisibleContext value={placement.visible}>
            <ContribBoundary id={pane.id} key={epoch}>
              {pane.render && <ContribRender render={pane.render} />}
            </ContribBoundary>
          </PaneVisibleContext>
        </PaneLifecycleContext>
      </PaneGroupContext>
    </div>
  )
})

export const useStablePaneHosts = () => useContext(PlacementContext) !== null

interface KeepAlivePaneSlotProps extends Omit<Placement, 'anchor'> {
  paneId: string
}

/** The tree owns placement, never the body. Slots may freely unmount or move. */
export function KeepAlivePaneSlot({
  paneId,
  groupId,
  headerVisible,
  visible,
  overlay,
  onMouseLeave
}: KeepAlivePaneSlotProps) {
  const place = useContext(PlacementContext)
  const sides = useStore($collapsedTreeSides)
  const editing = useStore($layoutEditMode)
  const side = paneRootSide(paneId)
  const shown = visible && (overlay || editing || side === null || !sides.has(side))
  const id = useId()
  const anchor = `--pane-${id.replace(/[^a-zA-Z0-9_-]/g, '-')}`

  useLayoutEffect(
    () => place?.(paneId, { anchor, groupId, headerVisible, visible: shown, overlay, onMouseLeave }),
    [place, paneId, anchor, groupId, headerVisible, shown, overlay, onMouseLeave]
  )

  return (
    <div aria-hidden className="pointer-events-none absolute inset-0" style={{ anchorName: anchor } as CSSProperties} />
  )
}
