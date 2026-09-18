import { type Codec, persistentAtom } from '@/lib/persisted'

export const IN_APP_TOAST_CORNERS = ['top-left', 'top-right', 'bottom-left', 'bottom-right'] as const

export type InAppToastCorner = (typeof IN_APP_TOAST_CORNERS)[number]

export const DEFAULT_IN_APP_TOAST_CORNER: InAppToastCorner = 'bottom-right'

const STORAGE_KEY = 'hermes.desktop.inAppToastCorner'
const corners = new Set<string>(IN_APP_TOAST_CORNERS)

export function resolveInAppToastCorner(value: unknown): InAppToastCorner {
  return typeof value === 'string' && corners.has(value) ? (value as InAppToastCorner) : DEFAULT_IN_APP_TOAST_CORNER
}

const codec: Codec<InAppToastCorner> = {
  decode: resolveInAppToastCorner,
  encode: value => value
}

export const $inAppToastCorner = persistentAtom(STORAGE_KEY, DEFAULT_IN_APP_TOAST_CORNER, codec)

export function setInAppToastCorner(corner: InAppToastCorner) {
  $inAppToastCorner.set(resolveInAppToastCorner(corner))
}
