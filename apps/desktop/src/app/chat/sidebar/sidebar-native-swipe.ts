export type NativeSwipeDirection = 'left' | 'right'

export interface PointerPosition {
  clientX: number
  clientY: number
}

export interface RegionBounds {
  bottom: number
  left: number
  right: number
  top: number
}

interface NativeSwipeActions {
  cycleProfile: (direction: -1 | 1) => void
  navigatePreview: (command: 'back' | 'forward') => void
}

/** Geometry remains reliable when the hover chain is stale after a render. */
export function isPointerInsideBounds(pointer: PointerPosition, bounds: RegionBounds): boolean {
  return (
    pointer.clientX >= bounds.left &&
    pointer.clientX <= bounds.right &&
    pointer.clientY >= bounds.top &&
    pointer.clientY <= bounds.bottom
  )
}

/** Give the hovered sidebar ownership; preserve preview navigation elsewhere. */
export function routeNativeSwipe(
  direction: NativeSwipeDirection,
  sidebarHovered: boolean,
  actions: NativeSwipeActions
): 'profile' | 'preview' {
  if (sidebarHovered) {
    actions.cycleProfile(direction === 'left' ? -1 : 1)

    return 'profile'
  }

  actions.navigatePreview(direction === 'left' ? 'back' : 'forward')

  return 'preview'
}
