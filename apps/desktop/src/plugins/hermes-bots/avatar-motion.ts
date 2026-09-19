/**
 * Blobatar motion — the bridge between Bot Mode's mood system and the
 * animation layer the blobatar library already ships.
 *
 * The library draws idle motion (blink, saccade gaze, breathe, bob) as PURE
 * CSS: one `motion.css` stylesheet plus per-face custom properties (seeded
 * phase offsets so a roster never breathes in unison), classes on the `<svg>`,
 * and zero JS per frame. It also carries an expression roster — including
 * `thinking`, the two-dot loader every user already reads — that maps cleanly
 * onto Bot Mode's existing idle/think/work mood.
 *
 * Until now the app called only the static string renderer (`blobatar()`),
 * which emits neither the classes nor the vars: every blob avatar in the
 * roster was a still frame. This module resolves the animated component and
 * the expression poses feature-first: any piece missing (older bundle, older
 * SDK) falls back to the existing static render — identical face, no motion.
 */
import type { ComponentType, CSSProperties } from 'react'

import type { FaceMood } from './types'

export type BlobatarProps = {
  name: string
  size?: number
  animate?: 'always' | 'hover'
  expression?: unknown
  className?: string
  style?: CSSProperties
  [key: string]: unknown
}

type BlobatarModule = {
  Blobatar: ComponentType<BlobatarProps>
  expressions: Record<string, unknown> | null
}

let cached: BlobatarModule | null | undefined

/** Feature-detected once: the animated component + the expression roster.
 *  Null = this build of the SDK predates the motion layer; every caller then
 *  renders the static path unchanged. */
async function loadBlobatarMotion(): Promise<BlobatarModule | null> {
  if (cached !== undefined) {
    return cached
  }

  try {
    // Dynamic import (not require): the renderer is an ESM vite bundle, where
    // require() is undefined at runtime. Both subpaths are published by
    // blobatar 2.0.0 and the import graph resolves them at build time; the
    // try/catch only guards an SDK that stops publishing them.
    const [react, expression] = await Promise.all([
      import('blobatar/react'),
      import('blobatar/expression')
    ])

    const BlobatarComponent = react?.Blobatar as ComponentType<BlobatarProps> | undefined

    cached =
      BlobatarComponent && expression
        ? { Blobatar: BlobatarComponent, expressions: expression as Record<string, unknown> }
        : null
  } catch {
    cached = null
  }

  return cached
}

/** True when the running bundle carries the motion layer (classes + vars +
 *  expressions). Tests and callers use this to pin which path rendered. */
export async function blobatarMotionAvailable(): Promise<boolean> {
  return (await loadBlobatarMotion()) !== null
}

/** Bot mood -> blobatar expression name. `idle` keeps the plain idle loop
 *  (ambient breathe/blink IS the idle personality); `think` wears the
 *  library's `thinking` seesaw — the built-in two-dot loader; `work` wears
 *  `happy`. */
export function mapBotMoodToExpression(mood: FaceMood | string): string | undefined {
  if (mood === 'think') {
    return 'thinking'
  }

  if (mood === 'work') {
    return 'happy'
  }

  return undefined
}

/** The expression object for a mood, or undefined when unavailable — the
 *  caller then renders the animated face without a pose (still breathing,
 *  blinking and glancing: the idle loop needs no expression). */
export async function blobatarExpressionFor(mood: FaceMood | string): Promise<unknown> {
  const mod = await loadBlobatarMotion()

  if (!mod?.expressions) {
    return undefined
  }

  const name = mapBotMoodToExpression(mood)

  return name ? mod.expressions[name] : undefined
}

/** The animated Blobatar component, or null on an SDK without the motion
 *  layer (callers fall back to the static string renderer). */
export async function blobatarMotionComponent(): Promise<ComponentType<BlobatarProps> | null> {
  return (await loadBlobatarMotion())?.Blobatar ?? null
}
