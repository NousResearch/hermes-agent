import type { ReactNode } from 'react'

/** The component gallery's groups, in the order the page shows them. */
export const GALLERY_GROUPS = ['actions', 'inputs', 'overlays', 'feedback', 'layout', 'text', 'icons'] as const

export type GalleryGroup = (typeof GALLERY_GROUPS)[number]

export interface GalleryExample {
  /** What this example shows: a variant, a size, a state. */
  label: string
  render: () => ReactNode
}

/**
 * One primitive on the design system page. `source` is the file it lives in, relative to `src/`; for a
 * `components/ui/` file that is what the coverage test counts as shown.
 */
export interface GalleryEntry {
  title: string
  source: string
  /** One line: what to reach for it for. */
  use: string
  group: GalleryGroup
  examples: readonly GalleryExample[]
  /** Other files these examples render (a recipe or a sibling primitive), counted as shown too. */
  covers?: readonly string[]
  /** Many small examples (an icon set): a tight grid. */
  dense?: boolean
}

/**
 * Every member of a union, checked at compile time: `everyOf<Variant>()('a', 'b')` fails to type-check
 * until each variant is listed, so a variant added to a primitive is a gallery error, not a gap.
 */
export const everyOf =
  <T extends string>() =>
  <const V extends readonly T[]>(...values: V & ([Exclude<T, V[number]>] extends [never] ? unknown : never)): V =>
    values
