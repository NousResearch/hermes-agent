import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import type * as ProfileIdGuard from './profile-id-guard'

// Every Electron profile-id gate takes its grammar from PROFILE_ID_ARG_RE (state-24, #155595).
// Swapping the owner for a different grammar shows each gate follows it; a re-declared copy of
// `^[a-z0-9][a-z0-9_-]{0,63}$` keeps accepting `work` and fails here.
vi.mock('./profile-id-guard', async importOriginal => ({
  ...(await importOriginal<typeof ProfileIdGuard>()),
  PROFILE_ID_ARG_RE: /^zz\.[a-z]+$/
}))

const OLD_ONLY = 'work'
const NEW_ONLY = 'zz.custom'

describe('profile-id gates follow PROFILE_ID_ARG_RE', () => {
  it('desktop routes and the launch flag', async () => {
    const { DESKTOP_PROFILE_NAME_RE } = await import('./desktop-profile')
    const { parseLaunchProfile } = await import('./launch-profile')

    expect(DESKTOP_PROFILE_NAME_RE.test(NEW_ONLY)).toBe(true)
    expect(DESKTOP_PROFILE_NAME_RE.test(OLD_ONLY)).toBe(false)
    expect(parseLaunchProfile(['--profile', NEW_ONLY])).toBe(NEW_ONLY)
    expect(parseLaunchProfile(['--profile', OLD_ONLY])).toBeNull()
  })

  it('local skin profile key', async () => {
    const { localSkinProfileKey } = await import('./local-skin')

    expect(localSkinProfileKey(NEW_ONLY)).toBe(NEW_ONLY)
    expect(localSkinProfileKey(OLD_ONLY)).toBe('default')
  })

  it('boot preference writer', async () => {
    const { writeDesktopProfile } = await import('./desktop-boot-preference')
    const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'profile-id-guard-')), 'boot.json')

    expect(writeDesktopProfile(file, NEW_ONLY)).toBe(NEW_ONLY)
    expect(() => writeDesktopProfile(file, OLD_ONLY)).toThrow('Invalid profile name')
  })

  it('saved SSH and URL-remote profile mappings', async () => {
    const { normalizeRemoteProfileName, normalizeSshConfig } = await import('./connection-config')

    expect(normalizeSshConfig({ mode: 'ssh', host: 'h', remoteProfile: NEW_ONLY })?.remoteProfile).toBe(NEW_ONLY)
    expect(normalizeSshConfig({ mode: 'ssh', host: 'h', remoteProfile: OLD_ONLY })?.remoteProfile).toBeUndefined()
    expect(normalizeRemoteProfileName(NEW_ONLY)).toEqual({ remoteProfile: NEW_ONLY })
    expect(normalizeRemoteProfileName(OLD_ONLY)).toBeNull()
  })
})
