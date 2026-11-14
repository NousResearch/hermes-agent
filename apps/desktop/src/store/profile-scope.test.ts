import { beforeEach, describe, expect, it } from 'vitest'

import {
  $activeGatewayProfile,
  $profiles,
  $profileScope,
  $showAllProfiles,
  ALL_PROFILES
} from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

// #101642: a single-profile install could get stuck with showAllProfiles=true
// (Grouping → Profile, or persisted from a two-profile era) with no rail
// control showing or clearing it — and every projects.* mutation throwing
// while the scope read ALL. The scope now ignores the flag at exactly one
// profile, matching the sidebar's own rendering exemption.

const profile = (name: string, isDefault = false): ProfileInfo => ({
  has_env: false,
  is_default: isDefault,
  model: null,
  name,
  path: `/tmp/${name}`,
  provider: null,
  skill_count: 0
})

describe('$profileScope single-profile exemption', () => {
  beforeEach(() => {
    window.localStorage.removeItem('hermes.desktop.showAllProfiles')
    $showAllProfiles.set(false)
    $activeGatewayProfile.set('default')
    $profiles.set([])
  })

  it('honors the all-profiles flag with multiple profiles', () => {
    $profiles.set([profile('default', true), profile('work')])
    $showAllProfiles.set(true)

    expect($profileScope.get()).toBe(ALL_PROFILES)
  })

  it('collapses ALL to the gateway profile at exactly one profile', () => {
    $profiles.set([profile('default', true)])
    $showAllProfiles.set(true)

    expect($profileScope.get()).toBe('default')
  })

  it('keeps honoring the flag before the profile list has loaded', () => {
    // An empty list means "not loaded yet", not "zero profiles" — collapsing
    // here would flicker a multi-profile boot profile → all as the list lands.
    $showAllProfiles.set(true)

    expect($profileScope.get()).toBe(ALL_PROFILES)
  })

  it('collapses when the profile count drops to one, restores at two', () => {
    $profiles.set([profile('default', true), profile('work')])
    $showAllProfiles.set(true)
    expect($profileScope.get()).toBe(ALL_PROFILES)

    $profiles.set([profile('default', true)])
    expect($profileScope.get()).toBe('default')

    // The flag stays persisted, so a second profile brings the view back.
    $profiles.set([profile('default', true), profile('work')])
    expect($profileScope.get()).toBe(ALL_PROFILES)
  })

  it('still follows the live gateway profile when the flag is off', () => {
    $profiles.set([profile('default', true), profile('work')])
    $activeGatewayProfile.set('work')

    expect($profileScope.get()).toBe('work')
  })
})
