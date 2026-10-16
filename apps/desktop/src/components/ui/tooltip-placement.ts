export type TooltipPlacement = 'control' | 'row' | 'toolbar' | 'left-rail' | 'right-rail'

/** Preferred sides express intent; Radix still flips and shifts for available space. */
export const TOOLTIP_PLACEMENTS = {
  control: { side: 'top', align: 'center' },
  row: { side: 'right', align: 'center' },
  toolbar: { side: 'bottom', align: 'center' },
  'left-rail': { side: 'right', align: 'center' },
  'right-rail': { side: 'left', align: 'center' }
} as const
