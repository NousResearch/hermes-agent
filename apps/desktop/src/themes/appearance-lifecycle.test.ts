import { afterEach, beforeEach, expect, it } from 'vitest'

import type { HermesConnection } from '@/global'
import { $connection } from '@/store/session'
import { dropTilesForProfile, migrateTilesForProfile } from '@/store/session-states'

import { modePref, skinPref } from './appearance-prefs'

beforeEach(() => {
  window.localStorage.clear()
})

afterEach(() => {
  $connection.set(null)
})

it('does not erase local appearance for a legacy remote without a stable id', () => {
  skinPref.assign('work', 'catppuccin', 'local')
  skinPref.assign('other', 'nous', 'local')
  $connection.set({ mode: 'remote', profile: 'default' } as HermesConnection)
  dropTilesForProfile('work')
  expect(skinPref.resolve('work', 'local')).toBe('catppuccin')
})

it('drops appearance only on the deleted gateway and masks legacy assignments there', () => {
  window.localStorage.setItem('hermes-desktop-profile-themes-v1', JSON.stringify({ work: 'midnight', other: 'ember' }))
  skinPref.assign('work', 'catppuccin', 'remote')
  modePref.assign('work', 'dark', 'remote')
  skinPref.assign('other', 'nous', 'remote')
  modePref.assign('other', 'light', 'remote')
  skinPref.assign('work', 'ember', 'local')
  window.localStorage.setItem(
    'hermes-desktop-active-appearance-v2',
    JSON.stringify({ connectionId: 'remote', profile: 'work' })
  )
  dropTilesForProfile('alias', { connectionId: 'remote', profile: 'alias', targetProfile: 'work' })
  expect([skinPref.resolve('work', 'remote'), modePref.resolve('work', 'remote')]).toEqual(['nous', 'light'])
  expect(skinPref.resolve('work', 'local')).toBe('ember')
  expect(skinPref.resolve('work', 'unvisited-peer')).toBe('midnight')
  expect(JSON.parse(window.localStorage.getItem('hermes-desktop-active-appearance-v2')!)).toEqual({
    connectionId: 'remote',
    profile: 'default'
  })
})

it('overwrites orphaned rename destinations with the source raw appearance', () => {
  skinPref.assign('work', 'late-backend-skin', 'local')
  modePref.assign('work', 'dark', 'local')
  skinPref.assign('studio', 'ember', 'local')
  modePref.assign('studio', 'light', 'local')
  skinPref.assign('studio', 'catppuccin', 'remote')
  migrateTilesForProfile('work', 'studio')
  expect(skinPref.stored('studio', 'local')).toBe('late-backend-skin')
  expect(modePref.resolve('studio', 'local')).toBe('dark')
  expect(skinPref.resolve('studio', 'remote')).toBe('catppuccin')
})

it('clears an orphaned rename destination when the source had no stored appearance', () => {
  window.localStorage.setItem('hermes-desktop-profile-themes-v2', JSON.stringify({ 'conn:local::studio': 'ember' }))
  window.localStorage.setItem('hermes-desktop-profile-modes-v2', JSON.stringify({ 'conn:local::studio': 'dark' }))
  migrateTilesForProfile('work', 'studio')
  expect(skinPref.stored('studio', 'local')).toBeNull()
  expect(modePref.resolve('studio', 'local')).toBe('system')
})

it('moves local appearance with the profile lifecycle without moving a remote namesake', () => {
  window.localStorage.setItem('hermes-desktop-profile-themes-v1', JSON.stringify({ work: 'midnight' }))
  skinPref.assign('work', 'ember', 'local')
  modePref.assign('work', 'dark', 'local')
  skinPref.assign('other', 'nous', 'local')
  modePref.assign('other', 'light', 'local')
  skinPref.assign('work', 'catppuccin', 'remote')
  modePref.assign('work', 'system', 'remote')
  window.localStorage.setItem(
    'hermes-desktop-active-appearance-v2',
    JSON.stringify({ connectionId: 'local', profile: 'work' })
  )
  migrateTilesForProfile('work', 'studio')
  expect([skinPref.resolve('studio', 'local'), modePref.resolve('studio', 'local')]).toEqual(['ember', 'dark'])
  expect([skinPref.resolve('work', 'local'), modePref.resolve('work', 'local')]).toEqual(['nous', 'light'])
  expect([skinPref.resolve('work', 'remote'), modePref.resolve('work', 'remote')]).toEqual(['catppuccin', 'system'])
  expect(JSON.parse(window.localStorage.getItem('hermes-desktop-active-appearance-v2')!)).toEqual({
    connectionId: 'local',
    profile: 'studio'
  })
  expect(JSON.parse(window.localStorage.getItem('hermes-desktop-profile-themes-v1')!)).toEqual({ work: 'midnight' })
})
