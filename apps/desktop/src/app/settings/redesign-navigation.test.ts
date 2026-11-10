import { expect, it, vi } from 'vitest'

import type { OverlayNavGroup, OverlayNavLink } from '../overlays/overlay-split-layout'

import { redesignSettingsNavigation } from './redesign-navigation'

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
