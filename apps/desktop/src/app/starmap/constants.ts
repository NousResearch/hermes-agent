import type { StarmapNode } from '@/types/hermes'

import type { GraphParams, Rgb, RingParams, Shape } from './types'

// ── Disk geometry ────────────────────────────────────────────────────────────
export const RING_INNER = 58
export const RING_OUTER = 340
export const ZOOM_MIN = 0.3
export const ZOOM_MAX = 5
export const FIT_PADDING = 80
export const TILT = 1 // vertical squash → "looking down at a tilted disk"
export const RING_STEPS = 4

export const WHITE: Rgb = { b: 255, g: 255, r: 255 }
export const BLACK: Rgb = { b: 0, g: 0, r: 0 }

// Fixed recency (age) gradient — old content quiet, recent content bright.
export const AGE_GRADIENT = { mid: 0.52, midInk: 0.74, newInk: 0.95, oldInk: 0.42, reach: 1 }

// Node glyph per kind — pure path geometry (the seam a future sprite/instanced
// renderer would bake from).
export const NODE_SHAPE: Record<StarmapNode['kind'], Shape> = { memory: 'diamond', skill: 'circle' }

// Darken the orb body so a bright primary doesn't swallow the sheen (the
// highlight is computed from the original ink, so it still reads).
export const ORB_DARKEN = 0.3

// Sheen forced this high when the orb ink is near-white (a white body needs a
// pure-white core to read as a sphere at all).
export const WHITEISH_SHEEN = 0.95

// Flat wash alpha for a lit (hovered/selected) date's band. The focused ring
// outline derives from this (×2).
export const LIT_BAND_ALPHA = 0.04

export const MODE_DEFAULTS: Record<'dark' | 'light', GraphParams> = {
  dark: {
    lineAlpha: 0.24,
    lineDash: 1.5,
    lineDashed: true,
    lineWidth: 0.5,
    ringAlpha: 0.1,
    ringDash: 4,
    ringDashed: false,
    ringWidth: 1.5
  },
  light: {
    lineAlpha: 0.18,
    lineDash: 1.5,
    lineDashed: true,
    lineWidth: 0.5,
    ringAlpha: 0.06,
    ringDash: 4,
    ringDashed: false,
    ringWidth: 2
  }
}

export const RING_PARAMS: Record<'dark' | 'light', RingParams> = {
  dark: { bandAlpha: 0.01, lightSize: 0.64, ringAlpha: 0.06, sheen: 0.12 },
  light: { bandAlpha: 0.03, lightSize: 0.27, ringAlpha: 0.028, sheen: 0.1 }
}
