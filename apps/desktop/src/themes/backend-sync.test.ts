import { beforeEach, describe, expect, it } from 'vitest'

import {
  $backendCustomCSS,
  $backendThemes,
  $pendingSkinApply,
  __resetBackendSkinSync,
  ingestBackendSkin,
} from './backend-sync'

const skin = (name: string) => ({
  name,
  colors: { background: '#101020', ui_accent: '#ff33aa', banner_text: '#eeeeee' }
})

const skinWithCSS = (name: string, customCSS: string) => ({ ...skin(name), customCSS })

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

  it('carries customCSS for a built-in-named user skin without shadowing the palette', () => {
    ingestBackendSkin(skinWithCSS('mono', '.chat-input { font-size: 16px; }'), { apply: true })

    expect($backendThemes.get().mono).toBeUndefined() // palette policy preserved
    expect($pendingSkinApply.get()).toBe('mono')
    expect($backendCustomCSS.get().mono).toBe('.chat-input { font-size: 16px; }')
  })

  it('keys default-named skin CSS under the resolved desktop default', () => {
    ingestBackendSkin(skinWithCSS('default', 'body { background: red; }'), { apply: true })

    expect($backendThemes.get().default).toBeUndefined()
    // setTheme normalizes `default` → DEFAULT_SKIN_NAME ('nous'), so the CSS
    // must be findable under that name when the theme is derived.
    expect($backendCustomCSS.get().nous).toBe('body { background: red; }')
  })

  it('clears customCSS when a built-in-named skin drops the field', () => {
    ingestBackendSkin(skinWithCSS('mono', 'a { color: red; }'), { apply: true })
    expect($backendCustomCSS.get().mono).toBe('a { color: red; }')

    // Same skin, CSS removed from the YAML — the entry must go so stale rules
    // don't linger after the next apply.
    ingestBackendSkin(skin('mono'), { apply: true })

    expect($backendCustomCSS.get().mono).toBeUndefined()
  })

  it('does not populate the CSS store for non-built-in skins (converter carries it)', () => {
    ingestBackendSkin(skinWithCSS('neon', 'a { color: red; }'), { apply: true })

    expect($backendCustomCSS.get()).toEqual({})
    expect($backendThemes.get().neon?.customCSS).toBe('a { color: red; }')
  })

  it('reset clears the custom CSS store', () => {
    ingestBackendSkin(skinWithCSS('mono', 'a {}'), { apply: false })
    expect($backendCustomCSS.get().mono).toBe('a {}')

    __resetBackendSkinSync()

    expect($backendCustomCSS.get()).toEqual({})
  })

  it('ignores empty payloads', () => {
    ingestBackendSkin(undefined, { apply: true })
    ingestBackendSkin({ name: '' }, { apply: true })

    expect($pendingSkinApply.get()).toBeNull()
  })
})
