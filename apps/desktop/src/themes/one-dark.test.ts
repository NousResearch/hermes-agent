import { beforeEach, describe, expect, it } from 'vitest'

import { $backendThemes, __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'
import { isOneDark } from './one-dark'
import { nousTheme } from './presets'
import { skinToDesktopTheme } from './skin'
import { $userThemes, listAllThemes, resolveTheme } from './user-themes'

const legacy = (name = 'one-dark') => ({
  ...nousTheme,
  name,
  label: 'One-dark',
  colors: { ...nousTheme.colors, background: '#21252b', primary: '#c678dd' },
  darkColors: { ...nousTheme.colors, background: '#21252b', primary: '#c678dd' }
})

describe('One Dark cached theme resolution', () => {
  beforeEach(() => {
    __resetBackendSkinSync()
    $userThemes.set({})
    window.localStorage.clear()
  })

  it('repairs legacy backend snapshots before connect, without rewriting stored data', () => {
    const snapshot = legacy()
    $backendThemes.set({ 'one-dark': snapshot })
    const stored = window.localStorage.getItem('hermes-desktop-backend-themes-v1')
    const resolved = resolveTheme('one-dark')!
    expect(resolved.colors.background).toBe('#282c34')
    expect(isOneDark(resolved)).toBe(true)
    expect(listAllThemes().find(t => t.name === 'one-dark')!.colors).toEqual(resolved.colors)
    expect($backendThemes.get()['one-dark']).toBe(snapshot)
    expect(window.localStorage.getItem('hermes-desktop-backend-themes-v1')).toBe(stored)
  })

  it('rehydrates a generated saved accent from JSON without losing its pick', () => {
    const name = 'one-dark-accent-2ea043'
    const saved = { ...legacy(name), accentOf: 'one-dark', customCSS: '.mine { color: red; }' }
    $userThemes.set({ [name]: JSON.parse(JSON.stringify(saved)) })
    const resolved = resolveTheme(name)!
    expect(isOneDark(resolved)).toBe(true)
    expect(resolved.colors.primary).toBe('#404754')
    expect(resolved.colors.midground).not.toBe('#c678dd')
    expect(resolved.darkColors).toEqual(resolved.colors)
    expect(resolved.customCSS).toBe(saved.customCSS)
    expect(listAllThemes().find(t => t.name === name)!.colors).toEqual(resolved.colors)
    expect($userThemes.get()[name].colors.background).toBe('#21252b')
  })

  it('preserves user precedence, arbitrary names and inconsistent accent metadata', () => {
    ingestBackendSkin({ name: 'one-dark', colors: { ui_accent: '#c678dd' } }, { apply: false })
    const custom = legacy()
    const prefix = legacy('one-dark-something')
    const inconsistent = { ...legacy('one-dark-accent-2ea043'), accentOf: 'other' }
    $userThemes.set({ 'one-dark': custom, [prefix.name]: prefix, [inconsistent.name]: inconsistent })
    expect(resolveTheme('one-dark')).toBe(custom)
    expect(isOneDark(custom)).toBe(false)
    expect(resolveTheme(prefix.name)).toBe(prefix)
    expect(resolveTheme(inconsistent.name)).toBe(inconsistent)
    // Even an explicitly installed copy of the backend palette is user-owned.
    const installed = JSON.parse(
      JSON.stringify(
        skinToDesktopTheme({
          name: 'one-dark',
          colors: { ui_accent: '#c678dd' }
        })
      )
    )
    $userThemes.set({ 'one-dark': installed })
    expect(isOneDark(resolveTheme('one-dark')!)).toBe(false)
  })
})
