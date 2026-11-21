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
    const original = flatten(source).find(item => item.id === link.id)!
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
  // Provider accounts are per-agent; they stay under their own group instead of leading the rail.
  expect(result.find(link => link.id === 'pview:accounts')).toBeUndefined()
  expect(result.find(link => link.id === 'providers')?.active).toBe(true)
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
    'billing',
    'config:appearance',
    'notifications',
    'keybinds',
    'gateway',
    'sessions',
    'about',
    'config:advanced'
  ])
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
  expect(redesigned.headerGroup?.id).toBe('keybinds')
  expect(redesigned.headerChild).toBeUndefined()
})

it('names destinations per the redesign while keeping advanced subpage headings', () => {
  const billing = group('billing', undefined, true)

  const presentation = settingsNavigationPresentation([billing], true, 'billing', undefined, {
    billing: 'Account & Plan'
  })

  expect(presentation.groups[0].label).toBe('Account & Plan')
  expect(presentation.headerGroup?.label).toBe('Account & Plan')
  expect(settingsNavigationPresentation([billing], false, 'billing', undefined, { billing: 'x' }).headerGroup).toBe(
    billing
  )

  const speaking = group('speaking', undefined, true)
  const voice = group('config:voice', [speaking], true)
  const source = [voice, group('config:advanced')]
  const presentation2 = settingsNavigationPresentation(source, true, 'config:voice', 'speaking')
  expect(presentation2.headerGroup).toBe(voice)
  expect(presentation2.headerChild).toBe(speaking)
  expect(presentation2.groups.find(item => item.id === 'config:advanced')?.active).toBe(true)
})

it('lands Account & Plan on the billing overview without a lone sub-page row or breadcrumb', () => {
  const overview = group('bview:overview', undefined, true)
  const billing = group('billing', [overview], true)
  const presentation = settingsNavigationPresentation([billing], true, 'billing', undefined)
  expect(presentation.groups[0].children).toBeUndefined()
  expect(presentation.headerChild).toBeUndefined()
})

it('draws one-page destinations without their own subpage rows but keeps related pages filed under them', () => {
  const connection = group('gateway:connection', undefined, true)
  const webAccess = group('web-access')
  const gateway = group('gateway', [connection], true)
  const about = group('about', [group('about:updates', undefined, true), group('about:uninstall')], true)

  const redesigned = settingsNavigationPresentation([gateway, webAccess, about], true, 'gateway', 'connection')
  const gatewayRow = redesigned.groups.find(item => item.id === 'gateway')
  expect(gatewayRow?.children?.map(child => child.id)).toEqual(['web-access'])
  expect(redesigned.groups.find(item => item.id === 'about')?.children).toBeUndefined()
  expect(redesigned.headerChild).toBeUndefined()

  const classic = settingsNavigationPresentation([gateway, webAccess, about], false, 'gateway', 'connection')
  expect(classic.groups.find(item => item.id === 'about')?.children).toHaveLength(2)
})
