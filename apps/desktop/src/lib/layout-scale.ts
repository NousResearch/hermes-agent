import { atom } from 'nanostores'

/**
 * How much larger than the browser's own pixels the app is drawn, when the app does the enlarging
 * itself (the web app's UI scale, a `zoom` on the page). 1 everywhere else: a native window's zoom
 * and the browser's own zoom both shrink the viewport the page is given, so media queries already
 * see the space that is left.
 *
 * A page zoomed by its own CSS keeps its viewport. At 200% the app has half the room, and every size
 * query still answers for the whole window: the desktop layout, crammed into a phone's width. So
 * size queries are asked against the room that is actually left: a query's lengths are multiplied
 * by this scale before they are compared with the window, which is the same comparison as dividing
 * the window by it. Stylesheets get the same treatment (app/redesign/layout-scale.ts).
 */
export const $layoutScale = atom(1)

const SIZE_FEATURE = /\(([^()]*\b(?:width|height)\b[^()]*)\)/g
const LENGTH = /(\d*\.?\d+)(px|rem|em)\b/g

/** `query` with the lengths of its width and height features multiplied by `scale`. Other features
 *  (pointer, hover, colour scheme, resolution) are left as written. */
export function scaleMediaQuery(query: string, scale: number): string {
  if (scale === 1) {
    return query
  }

  return query.replace(SIZE_FEATURE, feature =>
    feature.replace(LENGTH, (_, value: string, unit: string) => `${Number((Number(value) * scale).toFixed(4))}${unit}`)
  )
}

/** `query` as it has to be asked of this window right now. */
export const layoutQuery = (query: string): string => scaleMediaQuery(query, $layoutScale.get())
