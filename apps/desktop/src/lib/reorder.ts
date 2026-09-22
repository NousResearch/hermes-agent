/**
 * THE reorder feel — one primitive for every horizontal drag-to-reorder strip
 * (profile rail squares, pane tab chips, and whatever comes next). Reorder
 * surfaces must read identically:
 *
 *  - the dragged item GLIDES BETWEEN SNAPPED SLOTS (it steps cell-to-cell on
 *    the snappier drag transition, never floats freely),
 *  - displaced neighbors spring aside on the slower rail transition,
 *  - both use the same easeOutBack overshoot,
 *  - a haptic tick marks each slot crossing, a success pulse the commit.
 *
 * Consumers differ in machinery (dnd-kit for the profile rail, the layout
 * tree's pointer-capture drag for tabs) but share these exact parameters.
 */

import { triggerHaptic } from '@/lib/haptics'

/** easeOutBack — a little overshoot so items spring into their slot. */
export const REORDER_SPRING = 'cubic-bezier(0.34, 1.56, 0.64, 1)'

/** Displaced neighbors reflow on this (dnd-kit object + CSS string forms). */
export const REORDER_RAIL_DURATION_MS = 300
export const REORDER_RAIL_TRANSITION = { duration: REORDER_RAIL_DURATION_MS, easing: REORDER_SPRING }
export const REORDER_RAIL_TRANSITION_CSS = `transform ${REORDER_RAIL_DURATION_MS}ms ${REORDER_SPRING}`

/** The dragged item glides between snapped slots on this (snappier). */
export const REORDER_DRAG_TRANSITION_CSS = `transform 200ms ${REORDER_SPRING}`

/** Tick each time the drag crosses into a new slot. */
export const reorderStepHaptic = () => triggerHaptic('selection')

/** Satisfying confirm on a committed reorder. */
export const reorderCommitHaptic = () => triggerHaptic('success')
