export type PetAgentMotion = 'greet' | 'idle' | 'working'

interface PetAgentPickerAnchorRect {
  height: number
  viewportHeight?: number
  viewportWidth?: number
  width: number
  x: number
  y: number
}

export function petAgentMotion(
  hovered: boolean,
  activity: { busy?: boolean; reasoning?: boolean; toolRunning?: boolean }
): PetAgentMotion {
  if (activity.busy || activity.reasoning || activity.toolRunning) {
    return 'working'
  }

  return hovered ? 'greet' : 'idle'
}

/**
 * Open once from an explicit activation. Hover remains a visual greeting only,
 * so the common hover-then-click gesture cannot issue two show requests.
 */
export function openPetAgentPicker(
  show: ((mode: 'agents', anchorRect?: PetAgentPickerAnchorRect) => void) | undefined,
  pet: Element | null
): void {
  if (!show) {
    return
  }

  const bounds = pet?.getBoundingClientRect()

  const anchorRect = bounds
    ? {
        height: bounds.height,
        viewportHeight: window.innerHeight,
        viewportWidth: window.innerWidth,
        width: bounds.width,
        x: bounds.x,
        y: bounds.y
      }
    : undefined

  show('agents', anchorRect)
}
