import { atom } from 'nanostores'

import type { HermesBackgroundImage, HermesBackgroundResolveError, HermesBackgroundResolveResult } from '@/global'
import { readJson, storedBoolean, writeJson } from '@/lib/storage'

import { $activeGatewayProfile, normalizeProfileKey } from './profile'

export const BACKGROUND_INTERVALS = [15, 30, 60, 180, 360, 540, 720, 1_440] as const
export type DesktopBackgroundInterval = (typeof BACKGROUND_INTERVALS)[number]
export type DesktopBackgroundMode = 'folder' | 'hermes' | 'image' | 'none'

export interface DesktopBackgroundPreference {
  intervalMinutes: DesktopBackgroundInterval
  mode: DesktopBackgroundMode
  rotationAnchorMs: number
  shuffleSeed: number
  sourcePath: string | null
  strength: number
}

export interface DesktopBackgroundRuntime {
  current: HermesBackgroundImage | null
  error: HermesBackgroundResolveError | null
  previous: HermesBackgroundImage | null
  resolving: boolean
  sourceKey: string
  truncated: boolean
}

const STORAGE_KEY = 'hermes-desktop-profile-backgrounds-v1'
const LEGACY_BACKDROP_KEY = 'hermes.desktop.backdrop.v1'
const DEFAULT_INTERVAL: DesktopBackgroundInterval = 30
const DEFAULT_STRENGTH = 10
const CROSSFADE_MS = 400

const EMPTY_RUNTIME: DesktopBackgroundRuntime = {
  current: null,
  error: null,
  previous: null,
  resolving: false,
  sourceKey: '',
  truncated: false
}

const clampStrength = (value: unknown): number => {
  const number = Number(value)

  return Number.isFinite(number) ? Math.min(100, Math.max(0, Math.round(number))) : DEFAULT_STRENGTH
}

const normalizeInterval = (value: unknown): DesktopBackgroundInterval => {
  const number = Number(value)

  return BACKGROUND_INTERVALS.includes(number as DesktopBackgroundInterval)
    ? (number as DesktopBackgroundInterval)
    : DEFAULT_INTERVAL
}

const normalizeMode = (value: unknown): DesktopBackgroundMode | null =>
  value === 'folder' || value === 'hermes' || value === 'image' || value === 'none' ? value : null

export function normalizeBackgroundPreference(value: unknown): DesktopBackgroundPreference | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const candidate = value as Partial<DesktopBackgroundPreference>
  const mode = normalizeMode(candidate.mode)

  if (!mode) {
    return null
  }

  const sourcePath =
    typeof candidate.sourcePath === 'string' && candidate.sourcePath.trim() ? candidate.sourcePath.trim() : null

  if ((mode === 'folder' || mode === 'image') && !sourcePath) {
    return null
  }

  return {
    intervalMinutes: normalizeInterval(candidate.intervalMinutes),
    mode,
    rotationAnchorMs:
      Number.isFinite(candidate.rotationAnchorMs) && Number(candidate.rotationAnchorMs) >= 0
        ? Number(candidate.rotationAnchorMs)
        : 0,
    shuffleSeed: Number.isFinite(candidate.shuffleSeed) ? Number(candidate.shuffleSeed) >>> 0 : 0,
    sourcePath,
    strength: clampStrength(candidate.strength)
  }
}

function readPreferenceRecord(): Record<string, DesktopBackgroundPreference> {
  const parsed = readJson<Record<string, unknown>>(STORAGE_KEY)

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(parsed).flatMap(([profile, value]) => {
      const preference = normalizeBackgroundPreference(value)

      return preference ? [[profile, preference]] : []
    })
  )
}

const legacyDefault = (enabled = storedBoolean(LEGACY_BACKDROP_KEY, true)): DesktopBackgroundPreference => ({
  intervalMinutes: DEFAULT_INTERVAL,
  mode: enabled ? 'hermes' : 'none',
  rotationAnchorMs: 0,
  shuffleSeed: 0,
  sourcePath: null,
  strength: DEFAULT_STRENGTH
})

export function backgroundPreferenceForProfile(
  record: Record<string, DesktopBackgroundPreference>,
  profile: string,
  legacyEnabled = true
): DesktopBackgroundPreference {
  return record[profile] ?? record.default ?? legacyDefault(legacyEnabled)
}

function resolveProfilePreference(profile: string): DesktopBackgroundPreference {
  return backgroundPreferenceForProfile(readPreferenceRecord(), profile, storedBoolean(LEGACY_BACKDROP_KEY, true))
}

const currentProfile = () => normalizeProfileKey($activeGatewayProfile.get())

export const $backgroundPreference = atom<DesktopBackgroundPreference>(resolveProfilePreference(currentProfile()))
export const $backgroundRuntime = atom<DesktopBackgroundRuntime>(EMPTY_RUNTIME)

function randomSeed(): number {
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    return crypto.getRandomValues(new Uint32Array(1))[0]
  }

  return Math.floor(Math.random() * 0x1_0000_0000) >>> 0
}

function assignPreference(next: DesktopBackgroundPreference): void {
  const profile = currentProfile()
  const record = readPreferenceRecord()
  writeJson(STORAGE_KEY, { ...record, [profile]: next })
  $backgroundPreference.set(next)
}

export function setBackgroundMode(mode: Extract<DesktopBackgroundMode, 'hermes' | 'none'>): void {
  const current = $backgroundPreference.get()
  assignPreference({ ...current, mode, sourcePath: null })
}

export function setBackgroundSource(
  mode: Extract<DesktopBackgroundMode, 'folder' | 'image'>,
  sourcePath: string
): void {
  const current = $backgroundPreference.get()
  assignPreference({
    ...current,
    mode,
    rotationAnchorMs: Date.now(),
    shuffleSeed: randomSeed(),
    sourcePath
  })
}

export function setBackgroundInterval(intervalMinutes: DesktopBackgroundInterval): void {
  const current = $backgroundPreference.get()
  assignPreference({
    ...current,
    intervalMinutes: normalizeInterval(intervalMinutes),
    rotationAnchorMs: Date.now()
  })
}

export function setBackgroundStrength(strength: number): void {
  const current = $backgroundPreference.get()
  assignPreference({ ...current, strength: clampStrength(strength) })
}

// Small deterministic PRNG: every open window derives the same deck from the
// persisted seed + wall-clock deck number without coordinating timers.
function mulberry32(seed: number): () => number {
  let state = seed >>> 0

  return () => {
    state += 0x6d2b79f5
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)

    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000
  }
}

function rawShuffledDeck(images: HermesBackgroundImage[], seed: number, deckIndex: number): HermesBackgroundImage[] {
  const deck = [...images].sort((left, right) => left.id.localeCompare(right.id))
  const random = mulberry32((seed ^ Math.imul(deckIndex + 1, 0x9e3779b1)) >>> 0)

  for (let index = deck.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1))

    ;[deck[index], deck[swap]] = [deck[swap], deck[index]]
  }

  return deck
}

function shuffledDeck(images: HermesBackgroundImage[], seed: number, deckIndex: number): HermesBackgroundImage[] {
  // With two items, changing the deck order can only make the previous deck's
  // last item become the next deck's first. Keep one seeded order instead.
  if (images.length === 2) {
    return rawShuffledDeck(images, seed, 0)
  }

  const deck = rawShuffledDeck(images, seed, deckIndex)

  if (deckIndex > 0 && deck.length > 1) {
    const previous = rawShuffledDeck(images, seed, deckIndex - 1)
    const previousLast = previous.at(-1)?.id

    if (deck[0]?.id === previousLast) {
      ;[deck[0], deck[1]] = [deck[1], deck[0]]
    }
  }

  return deck
}

export function carouselOrder(
  images: HermesBackgroundImage[],
  preference: DesktopBackgroundPreference,
  now = Date.now()
): HermesBackgroundImage[] {
  if (images.length <= 1) {
    return images
  }

  const intervalMs = preference.intervalMinutes * 60_000
  const elapsed = Math.max(0, now - preference.rotationAnchorMs)
  const slot = Math.floor(elapsed / intervalMs)
  const deckIndex = Math.floor(slot / images.length)
  const position = slot % images.length
  const deck = shuffledDeck(images, preference.shuffleSeed, deckIndex)

  return [...deck.slice(position), ...deck.slice(0, position)]
}

function sourceKey(preference: DesktopBackgroundPreference): string {
  return `${currentProfile()}:${preference.mode}:${preference.sourcePath ?? ''}`
}

let resolveGeneration = 0
let rotationTimer: number | null = null
let previousTimer: number | null = null

const clearRotationTimer = () => {
  if (rotationTimer !== null) {
    window.clearTimeout(rotationTimer)
    rotationTimer = null
  }
}

function scheduleRotation(preference: DesktopBackgroundPreference): void {
  clearRotationTimer()

  if (preference.mode !== 'folder') {
    return
  }

  const intervalMs = preference.intervalMinutes * 60_000
  const elapsed = Math.max(0, Date.now() - preference.rotationAnchorMs)
  const nextBoundary = preference.rotationAnchorMs + (Math.floor(elapsed / intervalMs) + 1) * intervalMs
  rotationTimer = window.setTimeout(() => void refreshBackgroundRuntime(), Math.max(250, nextBoundary - Date.now()))
}

function preloadImage(image: HermesBackgroundImage): Promise<boolean> {
  return new Promise(resolve => {
    const element = new Image()
    element.onload = () => resolve(true)
    element.onerror = () => resolve(false)
    element.src = image.url
  })
}

async function firstLoadable(images: HermesBackgroundImage[]): Promise<HermesBackgroundImage | null> {
  for (const image of images) {
    if (await preloadImage(image)) {
      return image
    }
  }

  return null
}

export async function resolveBackgroundSource(
  mode: Extract<DesktopBackgroundMode, 'folder' | 'image'>,
  sourcePath: string
): Promise<HermesBackgroundResolveResult> {
  return window.hermesDesktop.background.resolve({ kind: mode, sourcePath })
}

export async function refreshBackgroundRuntime(): Promise<void> {
  if (typeof window === 'undefined' || !window.hermesDesktop?.background) {
    return
  }

  const preference = $backgroundPreference.get()
  const key = sourceKey(preference)
  const generation = ++resolveGeneration
  scheduleRotation(preference)

  if (preference.mode === 'hermes' || preference.mode === 'none' || !preference.sourcePath) {
    $backgroundRuntime.set({ ...EMPTY_RUNTIME, sourceKey: key })

    return
  }

  const previousState = $backgroundRuntime.get()
  $backgroundRuntime.set({
    ...previousState,
    error: null,
    resolving: true,
    sourceKey: key
  })

  let result: HermesBackgroundResolveResult

  try {
    result = await resolveBackgroundSource(preference.mode, preference.sourcePath)
  } catch {
    result = { error: 'unreadable', images: [], sourcePath: preference.sourcePath, truncated: false }
  }

  if (generation !== resolveGeneration || sourceKey($backgroundPreference.get()) !== key) {
    return
  }

  if (result.error || result.images.length === 0) {
    $backgroundRuntime.set({
      ...(previousState.sourceKey === key ? previousState : EMPTY_RUNTIME),
      error: result.error ?? 'empty',
      resolving: false,
      sourceKey: key,
      truncated: result.truncated
    })

    return
  }

  const ordered = preference.mode === 'folder' ? carouselOrder(result.images, preference) : result.images
  const selected = await firstLoadable(ordered)

  if (generation !== resolveGeneration || sourceKey($backgroundPreference.get()) !== key) {
    return
  }

  if (!selected) {
    $backgroundRuntime.set({
      ...(previousState.sourceKey === key ? previousState : EMPTY_RUNTIME),
      error: 'unreadable',
      resolving: false,
      sourceKey: key,
      truncated: result.truncated
    })

    return
  }

  const currentState = $backgroundRuntime.get()

  const unchanged =
    currentState.current?.id === selected.id && currentState.current.fingerprint === selected.fingerprint

  if (unchanged) {
    $backgroundRuntime.set({ ...currentState, error: null, resolving: false, truncated: result.truncated })

    return
  }

  if (previousTimer !== null) {
    window.clearTimeout(previousTimer)
  }

  $backgroundRuntime.set({
    current: selected,
    error: null,
    previous: currentState.sourceKey === key ? currentState.current : null,
    resolving: false,
    sourceKey: key,
    truncated: result.truncated
  })

  previousTimer = window.setTimeout(() => {
    const latest = $backgroundRuntime.get()

    if (latest.current?.url === selected.url) {
      $backgroundRuntime.set({ ...latest, previous: null })
    }
  }, CROSSFADE_MS + 50)
}

if (typeof window !== 'undefined') {
  $activeGatewayProfile.subscribe(profile => {
    $backgroundPreference.set(resolveProfilePreference(normalizeProfileKey(profile)))
  })

  $backgroundPreference.subscribe(() => void refreshBackgroundRuntime())

  window.addEventListener('storage', event => {
    if (event.key === STORAGE_KEY) {
      $backgroundPreference.set(resolveProfilePreference(currentProfile()))
    }
  })

  window.addEventListener('focus', () => void refreshBackgroundRuntime())
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      void refreshBackgroundRuntime()
    }
  })
  window.hermesDesktop?.onPowerResume?.(() => void refreshBackgroundRuntime())
}
