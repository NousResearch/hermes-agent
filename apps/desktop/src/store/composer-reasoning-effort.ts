import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

// The old effort cache mixes config and runtime reports with user picks. Its
// provenance is unknowable, so only new deliberate picks enter this store.
const STORAGE_KEY = 'hermes.desktop.composer.reasoning-override.v1'
let scope: string | null = '.registry.local.default'
let scopeProfile = 'default'
let generation = 0

export const getComposerReasoningScope = (): string | null => scope
export const getComposerReasoningGeneration = (): number => generation
const storedEffort = (): string => (scope === null ? '' : (storedString(STORAGE_KEY + scope) ?? ''))
export const $composerReasoningEffort = atom(storedEffort())
// Persistence is best-effort; Send uses the intent already painted in this window.
export const getComposerReasoningEffort = (): string => $composerReasoningEffort.get()

export function rescopeComposerReasoningEffort(nextScope: string | null, profile = 'default'): void {
  if (scope === nextScope) {
    return
  }

  scope = nextScope
  scopeProfile = profile
  generation += 1
  $composerReasoningEffort.set(storedEffort())
}

export function setComposerReasoningEffort(effort: string): void {
  if (scope === null) {
    return
  }

  generation += 1
  persistString(STORAGE_KEY + scope, effort || null)
  $composerReasoningEffort.set(effort)
}

if (typeof window !== 'undefined') {
  window.addEventListener('storage', event => {
    if (event.key === null || event.key === STORAGE_KEY + scope) {
      generation += 1
      $composerReasoningEffort.set(storedEffort())
    }
  })
}

export function migrateComposerReasoningForProfile(from: string, to: string): void {
  const oldScope = `.registry.local.${encodeURIComponent(from)}`
  const newScope = `.registry.local.${encodeURIComponent(to)}`
  const effort = storedString(STORAGE_KEY + oldScope)

  if (effort !== null) {
    persistString(STORAGE_KEY + newScope, effort)
    persistString(STORAGE_KEY + oldScope, null)
  }

  if (scope === oldScope) {
    rescopeComposerReasoningEffort(newScope, to)
  }
}

export function dropComposerReasoningForProfile(
  profile: string,
  route?: { connectionId?: string; profile?: string }
): void {
  const removedScope = route
    ? `.registry.${encodeURIComponent(route.connectionId?.trim() || 'local')}.${encodeURIComponent(route.profile?.trim() || profile)}`
    : scope === null
      ? null
      : scope.slice(0, -encodeURIComponent(scopeProfile).length) + encodeURIComponent(profile)

  if (!removedScope) {
    return
  }

  persistString(STORAGE_KEY + removedScope, null)

  if (removedScope === scope) {
    generation += 1
    $composerReasoningEffort.set('')
  }
}
