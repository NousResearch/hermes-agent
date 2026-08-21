import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $backendThemes, $pendingSkinApply, __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'

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

describe('backend skin persistence', () => {
  beforeEach(() => __resetBackendSkinSync())

  it('persists a backend skin to localStorage so it survives reload', () => {
    ingestBackendSkin(skin('neon'), { apply: false })

    // The memory store holds it AND localStorage got it, so at boot — before
    // any gateway event lands — `$backendThemes` rehydrates from storage and
    // `resolveTheme('neon')` succeeds.
    expect($backendThemes.get().neon?.name).toBe('neon')
    expect(window.localStorage.getItem('hermes-desktop-backend-skins-v1')).toContain('"neon"')
  })

  it('rehydrates the backend store from localStorage on a fresh module load', async () => {
    // Simulate a prior session having persisted a skin: seed the storage key
    // directly, then reload the module so the atom is re-constructed against
    // that storage — exactly what happens on a real cold start.
    const seeded = {
      neon: {
        name: 'neon',
        label: 'Neon',
        description: 'Hermes skin',
        colors: { background: '#000000', foreground: '#ffffff', primary: '#ff33aa' },
        darkColors: { background: '#000000', foreground: '#ffffff', primary: '#ff33aa' }
      }
    }

    window.localStorage.setItem('hermes-desktop-backend-skins-v1', JSON.stringify(seeded))

    vi.resetModules()
    const fresh = await import('./backend-sync')

    expect((fresh.$backendThemes.get().neon as unknown as { name?: string })?.name).toBe('neon')
  })

  it('drops unusable entries when rehydrating (corrupted / partial writes)', async () => {
    // A malformed seed must never shadow a live skin or a user install.
    const seeded = {
      broken: { name: 'broken', label: 'Broken', colors: {} }, // missing required color keys
      good: { name: 'good', label: 'Good', colors: { background: '#000', foreground: '#fff', primary: '#888' } }
    }

    window.localStorage.setItem('hermes-desktop-backend-skins-v1', JSON.stringify(seeded))

    vi.resetModules()
    const fresh = await import('./backend-sync')

    expect(fresh.$backendThemes.get().broken).toBeUndefined()
    expect((fresh.$backendThemes.get().good as unknown as { name?: string })?.name).toBe('good')
  })

  it('never rehydrates a built-in or default name from storage', async () => {
    // Built-ins and `default` are never persisted, so a stale/foreign entry
    // under those names must not come back either.
    const seeded = {
      default: { name: 'default', label: 'Default', colors: { background: '#000', foreground: '#fff', primary: '#888' } },
      mono: { name: 'mono', label: 'Mono', colors: { background: '#000', foreground: '#fff', primary: '#888' } }
    }

    window.localStorage.setItem('hermes-desktop-backend-skins-v1', JSON.stringify(seeded))

    vi.resetModules()
    const fresh = await import('./backend-sync')

    expect(fresh.$backendThemes.get().default).toBeUndefined()
    expect(fresh.$backendThemes.get().mono).toBeUndefined()
  })

  it('never persists default or built-ins', () => {
    ingestBackendSkin(skin('default'), { apply: true })
    ingestBackendSkin(skin('mono'), { apply: true })

    const raw = window.localStorage.getItem('hermes-desktop-backend-skins-v1') ?? ''
    expect(raw).not.toContain('"default"')
    expect(raw).not.toContain('"mono"')
  })

  it('does not rewrite storage when the skin is unchanged', () => {
    ingestBackendSkin(skin('neon'), { apply: false })
    const first = window.localStorage.getItem('hermes-desktop-backend-skins-v1')

    // Same skin, same payload → no store change → no localStorage rewrite.
    ingestBackendSkin(skin('neon'), { apply: true })

    expect(window.localStorage.getItem('hermes-desktop-backend-skins-v1')).toBe(first)
  })

  it('rewrites storage when the skin palette changes (in-place recolor)', () => {
    ingestBackendSkin(skin('neon'), { apply: false })
    const first = window.localStorage.getItem('hermes-desktop-backend-skins-v1')

    // A recolor of the same named skin changes the converted palette.
    ingestBackendSkin({ ...skin('neon'), colors: { background: '#202040', ui_accent: '#ff33aa', banner_text: '#eeeeee' } }, {
      apply: true
    })

    const second = window.localStorage.getItem('hermes-desktop-backend-skins-v1')
    expect(second).not.toBe(first)
    expect(second).toContain('"neon"')
  })
})
