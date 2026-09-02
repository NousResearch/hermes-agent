/**
 * Visual tokens for in-app browser comment mode.
 *
 * Drawn onto the GUEST page, which does not load our CSS variables, so these
 * are real colors — the same reason the agent overlay uses a hex. Codex's
 * comment chrome is a design-tool selection blue, not the app accent: the
 * outline has to read as "a reviewer marked this" on someone else's page.
 */

export const ANNOTATE_BLUE = '#2F80ED'
export const ANNOTATE_BLUE_FILL = 'rgba(47, 128, 237, 0.14)'
export const ANNOTATE_BLUE_RING = 'rgba(47, 128, 237, 0.35)'
export const ANNOTATE_MARKER_SIZE = 22
export const ANNOTATE_OUTLINE_WIDTH = '2px'
export const ANNOTATE_CROP_PAD = 12

/** Host comment pill sits on the guest page, so it uses real colors like the pin. */
export const ANNOTATE_PILL_BG = '#2A2A2A'
export const ANNOTATE_PILL_FG = '#F4F4F4'
export const ANNOTATE_PILL_SEND = '#3D3D3D'
export const ANNOTATE_CARD_WIDTH = 280
export const ANNOTATE_CARD_HEIGHT = 44

export const ANNOTATE_CSS_KEYS = [
  'color',
  'background-color',
  'font-size',
  'font-family',
  'font-weight',
  'line-height',
  'letter-spacing',
  'text-align',
  'display',
  'position',
  'width',
  'height',
  'padding',
  'margin',
  'border-radius',
  'opacity',
  'flex-direction',
  'gap',
  'justify-content',
  'align-items'
] as const

export type AnnotateCssKey = (typeof ANNOTATE_CSS_KEYS)[number]
