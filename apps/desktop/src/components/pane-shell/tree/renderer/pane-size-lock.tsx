import { useStore } from '@nanostores/react'
import type { RefObject } from 'react'

import { Codicon } from '@/components/ui/codicon'
import {
  $paneStates,
  type PaneStateSnapshot,
  setPaneHeightLocked,
  setPaneHeightOverride,
  setPaneWidthLocked,
  setPaneWidthOverride
} from '@/store/panes'

import { findParentSplit } from '../model'
import { $layoutTree } from '../store'

interface SizeLock {
  locked: boolean
  setLocked: (locked: boolean) => void
}

type LockAxis = 'height' | 'width'

function lockAxes(nodeId: string, shownCount: number): LockAxis[] {
  const tree = $layoutTree.get()
  const parent = tree ? findParentSplit(tree, nodeId) : null
  const columnParent = tree && parent?.orientation === 'column' ? findParentSplit(tree, parent.id) : null
  const localAxis = parent?.orientation === 'row' ? 'width' : parent?.orientation === 'column' ? 'height' : null
  const sharedColumnWidth = parent?.orientation === 'column' && columnParent?.orientation === 'row' && shownCount > 0
  const axes: LockAxis[] = []

  if (localAxis && parent && parent.children.length > 1 && shownCount > 0) {
    axes.push(localAxis)
  }

  if (sharedColumnWidth && columnParent && columnParent.children.length > 1) {
    // Only this zone owns the lock. Copying it into its siblings would
    // double-count the upper row and force the column wider.
    axes.push('width')
  }

  return axes
}

function makeSizeLock(
  axis: LockAxis,
  paneIds: string[],
  paneStates: Record<string, PaneStateSnapshot>,
  element: RefObject<HTMLDivElement | null>
): SizeLock {
  // A partially locked zone remains resizable; one click pins every tenant
  // instead of inverting the tenants independently.
  const locked = paneIds.length > 0 && paneIds.every(id => Boolean(paneStates[id]?.[axis === 'width' ? 'widthLocked' : 'heightLocked']))

  return {
    locked,
    setLocked: nextLocked => {
      const bounds = element.current?.getBoundingClientRect()

      if (!bounds) {
        return
      }

      const size = Math.round(axis === 'width' ? bounds.width : bounds.height)

      for (const paneId of paneIds) {
        if (axis === 'width') {
          if (nextLocked) {
            setPaneWidthOverride(paneId, size)
          }

          setPaneWidthLocked(paneId, nextLocked)
        } else {
          if (nextLocked) {
            setPaneHeightOverride(paneId, size)
          }

          setPaneHeightLocked(paneId, nextLocked)
        }
      }
    }
  }
}

export function usePaneSizeLock(nodeId: string, shown: string[], element: RefObject<HTMLDivElement | null>) {
  // A lock can be owned by any shown tenant, not just the active tab.
  const paneStates = useStore($paneStates)
  const sizeLocks = lockAxes(nodeId, shown.length).map(axis => makeSizeLock(axis, shown, paneStates, element))
  const canLock = sizeLocks.length > 0
  // The editor exposes one pane-level control, including for a column pane
  // whose local height and shared width are both lockable.
  const paneLocked = canLock && sizeLocks.every(sizeLock => sizeLock.locked)
  const paneLockLabel = paneLocked ? 'Pane is locked — click to unlock' : 'Pane is resizable — click to lock'
  const togglePaneLock = () => sizeLocks.forEach(sizeLock => sizeLock.setLocked(!paneLocked))

  return { canLock, paneLocked, paneLockLabel, togglePaneLock }
}

interface PaneSizeLockProps {
  element: RefObject<HTMLDivElement | null>
  nodeId: string
  shown: string[]
}

export function PaneSizeLock({ element, nodeId, shown }: PaneSizeLockProps) {
  const { canLock, paneLocked, paneLockLabel, togglePaneLock } = usePaneSizeLock(nodeId, shown, element)

  if (!canLock) {
    return null
  }

  return (
    <button
      aria-label={paneLockLabel}
      aria-pressed={paneLocked}
      className="-my-0.5 -mr-1 rounded p-1 text-(--ui-text-secondary) hover:bg-(--ui-bg-hover) hover:text-foreground focus-visible:outline-2 focus-visible:outline-(--ui-focus-border)"
      onClick={event => {
        event.stopPropagation()
        togglePaneLock()
      }}
      onPointerDown={event => event.stopPropagation()}
      type="button"
    >
      <Codicon name={paneLocked ? 'lock' : 'unlock'} size="0.8125rem" />
    </button>
  )
}
