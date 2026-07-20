/**
 * Avatar profile store — persisted to localStorage.
 *
 * Each participant (user / assistant) has a display name + optional
 * image data URL.  Names default to "You" / "Hermes".
 */

import { atom } from 'nanostores'

import { persistString, persistStringRecord, storedString, storedStringRecord } from '@/lib/storage'

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

const IMAGE_KEY = 'hermes.desktop.avatarImages.v1'
const NAME_KEY = 'hermes.desktop.avatarNames.v1'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type AvatarRole = 'user' | 'assistant'

export const DEFAULT_NAMES: Record<AvatarRole, string> = {
  assistant: 'Hermes',
  user: 'You'
}

// ---------------------------------------------------------------------------
// Image store  (Record<AvatarRole, dataURL | empty>)
// ---------------------------------------------------------------------------

function loadImages(): Record<AvatarRole, string> {
  const raw = storedStringRecord(IMAGE_KEY)
  const images: Record<string, string> = { assistant: '', user: '' }

  if (raw.assistant) images.assistant = raw.assistant
  if (raw.user) images.user = raw.user

  return images
}

export const $avatarImages = atom<Record<AvatarRole, string>>(loadImages())

$avatarImages.subscribe(images => persistStringRecord(IMAGE_KEY, images))

export function setAvatarImage(role: AvatarRole, dataUrl: string): void {
  const next = { ...$avatarImages.get(), [role]: dataUrl }

  $avatarImages.set(next)
}

// ---------------------------------------------------------------------------
// Name store  (Record<AvatarRole, string>)
// ---------------------------------------------------------------------------

function loadNames(): Record<AvatarRole, string> {
  const raw = storedStringRecord(NAME_KEY)
  const names = { ...DEFAULT_NAMES }

  if (raw.assistant) names.assistant = raw.assistant
  if (raw.user) names.user = raw.user

  return names
}

export const $avatarNames = atom<Record<AvatarRole, string>>(loadNames())

$avatarNames.subscribe(names => persistStringRecord(NAME_KEY, names))

export function setAvatarName(role: AvatarRole, name: string): void {
  const trimmed = name.trim()
  const fallback = DEFAULT_NAMES[role]

  // Empty → fall back to default, but still persist explicit empties as empty
  // so the store knows it was intentionally cleared.
  const next = { ...$avatarNames.get(), [role]: trimmed || fallback }

  $avatarNames.set(next)
}

/** Resolved name: returns the custom name or the default. */
export function getAvatarName(role: AvatarRole): string {
  return $avatarNames.get()[role] || DEFAULT_NAMES[role]
}

// ---------------------------------------------------------------------------
// Editor dialog open state
// ---------------------------------------------------------------------------

export const $avatarEditorOpen = atom<boolean>(false)

export function openAvatarEditor(): void {
  $avatarEditorOpen.set(true)
}

export function closeAvatarEditor(): void {
  $avatarEditorOpen.set(false)
}
