import { beforeEach, describe, expect, it } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'

import { $backendThemes, $pendingSkinApply, __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'
import { skinToDesktopTheme } from './skin'
import { $userThemes } from './user-themes'

const skin = (name: string) => ({
  name,
  colors: { background: '#101020', ui_accent: '#ff33aa', banner_text: '#eeeeee' }
})

describe('ingestBackendSkin', () => {
  beforeEach(() => {
    __resetBackendSkinSync()
    window.localStorage.clear()
    $activeGatewayProfile.set('default')
    $userThemes.set({})
  })

  it('registers a converted skin without applying when apply=false', () => {
    ingestBackendSkin(skin('neon'), { apply: false })

    expect($backendThemes.get().neon?.name).toBe('neon')
    expect($pendingSkinApply.get()).toBeNull()
  })

  it('applies a new skin name once', () => {
    ingestBackendSkin(skin('neon'), { apply: true })

    expect($pendingSkinApply.get()).toBe('neon')
  })

  it('does not re-apply the same skin name', () => {
    ingestBackendSkin(skin('neon'), { apply: true })
    $pendingSkinApply.set(null)
    ingestBackendSkin(skin('neon'), { apply: true })

    expect($pendingSkinApply.get()).toBeNull()
  })

  it('applies again when the skin name changes', () => {
    ingestBackendSkin(skin('neon'), { apply: true })
    $pendingSkinApply.set(null)
    ingestBackendSkin(skin('forest'), { apply: true })

    expect($pendingSkinApply.get()).toBe('forest')
  })

  it('seed does not paint, but a later same-name skin.changed applies (missed-activation recovery)', () => {
    // Connect while display.skin is already neon: seed records the baseline
    // without painting (never stomp the persisted desktop theme on connect).
    ingestBackendSkin(skin('neon'), { apply: false }) // gateway.ready seed
    expect($pendingSkinApply.get()).toBeNull()

    // The activation event was missed (skin set while disconnected / backend
    // restarted). Hermes re-affirms it — `hermes config set display.skin neon`
    // or a `hermes skin set` recolor. That explicit event must repaint even
    // though the name matches the seed.
    ingestBackendSkin(skin('neon'), { apply: true })
    expect($pendingSkinApply.get()).toBe('neon')

    // Once applied, a repeat same-name event is a no-op again...
    $pendingSkinApply.set(null)
    ingestBackendSkin(skin('neon'), { apply: true })
    expect($pendingSkinApply.get()).toBeNull()

    // ...and a genuine switch still applies.
    ingestBackendSkin(skin('forest'), { apply: true }) // Hermes authored a new skin
    expect($pendingSkinApply.get()).toBe('forest')
  })

  it('a reconnect re-seed after a real apply does not downgrade the applied baseline', () => {
    ingestBackendSkin(skin('neon'), { apply: true }) // applied for real
    $pendingSkinApply.set(null)

    ingestBackendSkin(skin('neon'), { apply: false }) // reconnect: gateway.ready re-seed
    ingestBackendSkin(skin('neon'), { apply: true }) // repeat event (e.g. in-place recolor)

    // Already painted once — the repeat must not re-apply (protects a manual
    // desktop-side theme switch from being snapped back after a reconnect).
    expect($pendingSkinApply.get()).toBeNull()
  })

  it('never registers default in the backend store (desktop keeps its own palette)', () => {
    ingestBackendSkin(skin('default'), { apply: true })

    expect($backendThemes.get().default).toBeUndefined()
  })

  it('does not apply default on the connect-time seed', () => {
    ingestBackendSkin(skin('default'), { apply: false })

    expect($pendingSkinApply.get()).toBeNull()
  })

  it('seed applies when the persisted desktop theme IS the backend skin (restart recovery)', () => {
    // User picked the backend-sourced skin last session; boot paint couldn't
    // resolve it before the gateway registered it, so it stored the name and
    // fell back to the default. The connect-time seed must finish that pick.
    window.localStorage.setItem('hermes-desktop-theme-v2', 'neon')
    ingestBackendSkin(skin('neon'), { apply: false })

    expect($pendingSkinApply.get()).toBe('neon')
  })

  it('seed applies a persisted per-profile backend skin', () => {
    window.localStorage.setItem('hermes-desktop-profile-themes-v1', JSON.stringify({ work: 'neon' }))
    $activeGatewayProfile.set('work')
    ingestBackendSkin(skin('neon'), { apply: false })

    expect($pendingSkinApply.get()).toBe('neon')
  })

  it('seed still does not apply when the persisted skin differs from the backend skin', () => {
    window.localStorage.setItem('hermes-desktop-theme-v2', 'mono')
    ingestBackendSkin(skin('neon'), { apply: false })

    expect($pendingSkinApply.get()).toBeNull()
  })

  it('persists a backend skin as a user theme so the next boot paint can resolve it', () => {
    ingestBackendSkin(skin('neon'), { apply: false })

    // Boot paint (before the gateway registers backend themes) resolves
    // built-ins and user themes from localStorage — so storing the converted
    // theme here makes a backend-sourced skin paint correctly on the very
    // first frame of the NEXT launch, not just after `gateway.ready`.
    expect($userThemes.get().neon?.name).toBe('neon')
    expect(window.localStorage.getItem('hermes-desktop-user-themes-v1')).toContain('"neon"')
  })

  it('does not rewrite a user theme that is already persisted unchanged', () => {
    const theme = skinToDesktopTheme(skin('neon'))!
    $userThemes.set({ neon: theme })
    window.localStorage.setItem('hermes-desktop-user-themes-v1', JSON.stringify({ neon: theme }))

    ingestBackendSkin(skin('neon'), { apply: false })

    expect($userThemes.get().neon?.name).toBe('neon')
  })

  it('applies a runtime switch back to default (repaints the desktop to its own default)', () => {
    ingestBackendSkin(skin('neon'), { apply: false }) // gateway.ready seed on some skin
    ingestBackendSkin(skin('default'), { apply: true }) // Hermes switched back to default

    expect($pendingSkinApply.get()).toBe('default')
  })

  it('does not shadow a built-in name but can still apply it', () => {
    ingestBackendSkin(skin('mono'), { apply: true })

    expect($backendThemes.get().mono).toBeUndefined()
    expect($pendingSkinApply.get()).toBe('mono')
  })

  it('ignores empty payloads', () => {
    ingestBackendSkin(undefined, { apply: true })
    ingestBackendSkin({ name: '' }, { apply: true })

    expect($pendingSkinApply.get()).toBeNull()
  })
})
