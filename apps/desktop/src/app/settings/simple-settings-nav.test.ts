import { expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'

import type { OverlayNavGroup } from '../overlays/overlay-split-layout'

import { groupSimpleSettings } from './simple-settings-nav'

it('organizes all destinations without losing handlers, nested routes, or optional pages', () => {
  const ids = [
    'config:appearance',
    'gateway',
    'capabilities',
    'config:model',
    'config:chat',
    'providers',
    'future-page'
  ]

  const groups: OverlayNavGroup[] = ids.map(id => ({
    id,
    label: id,
    active: id === 'providers',
    icon: () => null,
    onSelect: vi.fn()
  }))

  const nested = { id: 'providers:keys', label: 'API keys', active: true, icon: () => null, onSelect: vi.fn() }
  groups[5].children = [nested]
  const result = groupSimpleSettings(groups, en.settings.page)
  expect(new Set(result.map(g => g.id))).toEqual(new Set(ids))
  expect(result.slice(0, 3).map(g => g.id)).toEqual(['config:model', 'config:chat', 'capabilities'])
  expect(result.filter(g => g.heading).map(g => g.heading)).toEqual(['Agent', 'Connections', 'App'])
  expect(result.find(g => g.id === 'providers')?.children?.[0]).toBe(nested)
  result.find(g => g.id === 'providers')?.onSelect()
  expect(groups[5].onSelect).toHaveBeenCalledOnce()
  expect(result.at(-1)?.id).toBe('future-page')
})
