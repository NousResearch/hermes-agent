import { beforeEach, describe, expect, it } from 'vitest'

import { $backendThemes, $pendingSkinApply, __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'

const skin = (name: string) => ({ name, colors: { background: '#101020', ui_accent: '#ff33aa', banner_text: '#eeeeee' } })

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

  it('seeds on connect so the first matching poll is a no-op, but a change applies', () => {
    ingestBackendSkin(skin('neon'), { apply: false }) // gateway.ready seed
    ingestBackendSkin(skin('neon'), { apply: true }) // post-turn poll, unchanged
    expect($pendingSkinApply.get()).toBeNull()

    ingestBackendSkin(skin('forest'), { apply: true }) // Hermes authored a new skin
    expect($pendingSkinApply.get()).toBe('forest')
  })

  it('treats default as no-opinion: never registers or applies it', () => {
    ingestBackendSkin(skin('default'), { apply: true })

    expect($pendingSkinApply.get()).toBeNull()
    expect($backendThemes.get().default).toBeUndefined()
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
