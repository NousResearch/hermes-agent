import { expect, it, vi } from 'vitest'

import type { OverlayNavGroup, OverlayNavLink } from '../overlays/overlay-split-layout'

import { redesignSettingsNavigation, settingsNavigationPresentation } from './redesign-navigation'

const group = (id: string, children?: OverlayNavLink[], active = false): OverlayNavGroup => ({
  id,
  label: id,
  active,
  icon: () => null,
  onSelect: vi.fn(),
  children
})

const flatten = (groups: OverlayNavLink[]): OverlayNavLink[] =>
  groups.flatMap(group => [group, ...flatten(group.children ?? [])])

it('retains every destination, including conditional and unknown future settings', () => {
  const account = group('pview:accounts')
  const key = group('pview:keys')

  const source = [
    group('providers', [account, key]),
    group('keys', [group('kview:tools')]),
    group('config:appearance'),
    group('notifications', [group('sounds')]),
    group('gateway'),
    group('web-access'),
    group('config:safety'),
    group('vault'),
    group('config:advanced'),
    group('config:workspace'),
    group('future-setting')
  ]

  const rearranged = flatten(redesignSettingsNavigation(source))
  expect(rearranged.map(link => link.id).sort()).toEqual(
    flatten(source)
      .map(link => link.id)
      .sort()
  )

  for (const link of rearranged) {
    link.onSelect()
    const original = link.id === 'providers' ? key : flatten(source).find(item => item.id === link.id)!
    expect(original.onSelect).toHaveBeenCalled()
  }

  expect(source[0].children).toEqual([account, key])
})

it('highlights exactly the category containing the active child', () => {
  const result = redesignSettingsNavigation([
    group('config:appearance'),
    group('notifications', [group('sounds', undefined, true)], true),
    group('providers', [group('pview:accounts', undefined, true), group('pview:keys')], true)
  ])

  expect(result.find(link => link.id === 'notifications')?.active).toBe(true)
  expect(result.find(link => link.id === 'config:appearance')?.active).toBe(false)
  expect(result.find(link => link.id === 'providers')?.active).toBe(false)
  expect(result.find(link => link.id === 'pview:accounts')?.active).toBe(true)
})

it('keeps primary settings compact and exposes every advanced route within the three-level rail', () => {
  const result = redesignSettingsNavigation([
    group('providers', [group('pview:accounts'), group('pview:keys')]),
    group('billing', [group('billing:plans')]),
    group('config:appearance'),
    group('notifications'),
    group('keybinds'),
    group('gateway'),
    group('sessions'),
    group('about'),
    group('config:advanced', [group('advanced:runtime')]),
    group('config:voice', [group('voice:speaking')], true),
    group('plugins', [group('plugin:example', [group('plugin:example:details')])])
  ])
  expect(result.map(item => item.id)).toEqual([
    'pview:accounts',
    'config:appearance',
    'notifications',
    'keybinds',
    'gateway',
    'sessions',
    'about',
    'config:advanced'
  ])
  expect(result[0].children?.[0].id).toBe('billing')
  const advanced = result.find(item => item.id === 'config:advanced')!
  expect(advanced.active).toBe(true)
  expect(advanced.children?.find(item => item.id === 'plugins')?.children?.map(item => item.id)).toEqual([
    'plugin:example',
    'plugin:example:details'
  ])
  expect(advanced.children?.every(item => (item.children ?? []).every(child => !child.children))).toBe(true)
})

it('preserves classic navigation and subpage headings while simplifying redesigned landings', () => {
  const shortcuts = group('shortcuts', undefined, true)
  const keyboard = group('keybinds', [shortcuts], true)
  const source = [keyboard]
  const classic = settingsNavigationPresentation(source, false, 'keybinds', 'shortcuts')
  expect(classic.groups).toBe(source)
  expect(classic.headerGroup).toBe(keyboard)
  expect(classic.headerChild).toBe(shortcuts)

  const redesigned = settingsNavigationPresentation(source, true, 'keybinds', 'shortcuts')
  expect(redesigned.headerGroup).toBe(keyboard)
  expect(redesigned.headerChild).toBeUndefined()
})

it('titles the promoted account destination without losing advanced subpage headings', () => {
  const account = group('pview:accounts', undefined, true)
  const provider = group('providers', [account], true)
  expect(settingsNavigationPresentation([provider], true, 'providers', undefined).headerGroup).toBe(account)

  const speaking = group('speaking', undefined, true)
  const voice = group('config:voice', [speaking], true)
  const source = [voice, group('config:advanced')]
  const presentation = settingsNavigationPresentation(source, true, 'config:voice', 'speaking')
  expect(presentation.headerGroup).toBe(voice)
  expect(presentation.headerChild).toBe(speaking)
  expect(presentation.groups.find(item => item.id === 'config:advanced')?.active).toBe(true)
})
