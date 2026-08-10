import { beforeEach, describe, expect, it } from 'vitest'

import {
  $backendThemes,
  $pendingSkinApply,
  __BACKEND_THEMES_KEY,
  __resetBackendSkinSync,
  ingestBackendSkin
} from './backend-sync'

const skin = (name: string) => ({
  name,
  colors: { background: '#101020', ui_accent: '#ff33aa', banner_text: '#eeeeee' }
})

describe('ingestBackendSkin', () => {
  beforeEach(() => __resetBackendSkinSync())

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

// The desktop persists the active skin's NAME and resolves it against the theme
// registry on a window's first paint. A backend skin used to exist only in the
// connected renderer's memory, so a newly opened window (HUD, session pop-out)
// resolved that name to nothing and fell back to the default palette. Caching
// the CONVERTED theme is what lets a fresh renderer paint the right skin before
// — or without — its own gateway ever announcing it.
describe('converted backend skins survive a reload', () => {
  beforeEach(() => {
    window.localStorage.clear()
    __resetBackendSkinSync()
  })

  it('persists a registered skin so another window can resolve it cold', () => {
    ingestBackendSkin(skin('neon'), { apply: false })

    const cached = JSON.parse(window.localStorage.getItem(__BACKEND_THEMES_KEY) ?? '{}')

    expect(cached.neon?.name).toBe('neon')
    // A cached entry must be complete enough for the boot paint to use it —
    // the same bar `installUserTheme` holds a stored theme to.
    expect(typeof cached.neon?.colors?.background).toBe('string')
    expect(typeof cached.neon?.colors?.foreground).toBe('string')
    expect(typeof cached.neon?.colors?.primary).toBe('string')
  })

  it('a later switch replaces the cached palette rather than stacking a stale one', () => {
    ingestBackendSkin(skin('neon'), { apply: true })
    ingestBackendSkin({ ...skin('neon'), colors: { background: '#ffffff', ui_accent: '#101010' } }, { apply: true })

    const cached = JSON.parse(window.localStorage.getItem(__BACKEND_THEMES_KEY) ?? '{}')

    expect(cached.neon.colors.background).toBe('#ffffff')
  })

  // The backend only ever announces the ACTIVE skin, so nothing can refresh a
  // cached entry for any other one. Keeping them would strand a skin deleted
  // from disk in Appearance / `/skin` forever, offering a palette that no
  // longer exists — and grow the cache without bound as skins are authored.
  it('caches only the active skin, so a superseded one cannot linger', () => {
    ingestBackendSkin(skin('neon'), { apply: true })
    ingestBackendSkin(skin('forest'), { apply: true })

    const cached = JSON.parse(window.localStorage.getItem(__BACKEND_THEMES_KEY) ?? '{}')

    expect(Object.keys(cached)).toEqual(['forest'])
  })

  it('never caches a built-in name (the desktop palette stays authoritative)', () => {
    ingestBackendSkin(skin('mono'), { apply: true })

    expect(JSON.parse(window.localStorage.getItem(__BACKEND_THEMES_KEY) ?? '{}').mono).toBeUndefined()
  })
})
