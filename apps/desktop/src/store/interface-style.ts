import { atom } from 'nanostores'

import { readJson, writeJson } from '@/lib/storage'

export const INTERFACE_STYLE_STORAGE_KEY = 'hermes.desktop.interface-style.v1'

export interface InterfaceStylePreferences {
  controls: boolean
  enabled: boolean
  surfaces: boolean
  typography: boolean
}

export type InterfaceStylePart = Exclude<keyof InterfaceStylePreferences, 'enabled'>

export const DEFAULT_INTERFACE_STYLE: InterfaceStylePreferences = {
  controls: true,
  enabled: false,
  surfaces: true,
  typography: true
}

export function normalizeInterfaceStyle(value: unknown): InterfaceStylePreferences {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return { ...DEFAULT_INTERFACE_STYLE }
  }

  const record = value as Record<string, unknown>

  return {
    controls: typeof record.controls === 'boolean' ? record.controls : DEFAULT_INTERFACE_STYLE.controls,
    enabled: typeof record.enabled === 'boolean' ? record.enabled : DEFAULT_INTERFACE_STYLE.enabled,
    surfaces: typeof record.surfaces === 'boolean' ? record.surfaces : DEFAULT_INTERFACE_STYLE.surfaces,
    typography: typeof record.typography === 'boolean' ? record.typography : DEFAULT_INTERFACE_STYLE.typography
  }
}

const read = (): InterfaceStylePreferences =>
  typeof window === 'undefined'
    ? { ...DEFAULT_INTERFACE_STYLE }
    : normalizeInterfaceStyle(readJson<unknown>(INTERFACE_STYLE_STORAGE_KEY))

export const $interfaceStyle = atom<InterfaceStylePreferences>(read())

function commit(next: InterfaceStylePreferences): void {
  $interfaceStyle.set(next)
  writeJson(INTERFACE_STYLE_STORAGE_KEY, next)
}

export function setInterfaceStyleEnabled(enabled: boolean): void {
  commit({ ...$interfaceStyle.get(), enabled })
}

export function setInterfaceStylePart(part: InterfaceStylePart, enabled: boolean): void {
  commit({ ...$interfaceStyle.get(), [part]: enabled })
}

export function applyInterfaceStyleToDocument(
  preferences: InterfaceStylePreferences,
  root: HTMLElement = document.documentElement
): void {
  root.toggleAttribute('data-hermes-interface-style', preferences.enabled)
  root.toggleAttribute('data-hermes-interface-typography', preferences.enabled && preferences.typography)
  root.toggleAttribute('data-hermes-interface-controls', preferences.enabled && preferences.controls)
  root.toggleAttribute('data-hermes-interface-surfaces', preferences.enabled && preferences.surfaces)
}

/** Keep every renderer window in sync with the persisted Appearance preference. */
export function installInterfaceStyleSync(): () => void {
  const unsubscribe = $interfaceStyle.subscribe(preferences => applyInterfaceStyleToDocument(preferences))

  const onStorage = (event: StorageEvent) => {
    if (event.key !== INTERFACE_STYLE_STORAGE_KEY) {
      return
    }

    let value: unknown = null

    try {
      value = event.newValue === null ? null : JSON.parse(event.newValue)
    } catch {
      // A malformed external write resolves to the safe default.
    }

    $interfaceStyle.set(normalizeInterfaceStyle(value))
  }

  window.addEventListener('storage', onStorage)

  return () => {
    unsubscribe()
    window.removeEventListener('storage', onStorage)
  }
}
