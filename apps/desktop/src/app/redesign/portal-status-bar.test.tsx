import { describe, expect, it } from 'vitest'

import { STATUSBAR_WAITING, type StatusbarItem } from '@/app/shell/statusbar-controls'

import { statusBarItems, systemStatusItems } from './portal-status-bar'

const item = (id: string, extra: Partial<StatusbarItem> = {}): StatusbarItem => ({ id, label: id, ...extra })

const BAR = [
  item('command-center'),
  item('workspace-cwd'),
  item('gateway-health'),
  item('agents'),
  item('context-usage', { label: STATUSBAR_WAITING }),
  item('cache-hit-rate'),
  item('session-timer', { hidden: true }),
  item('approval-mode'),
  item('terminal'),
  item('version-client'),
  item('some-plugin:item')
]

const PLUGINS = new Set(['some-plugin:item'])

const ids = (items: readonly StatusbarItem[]) => items.map(entry => entry.id)

describe('the redesign’s split of the statusbar', () => {
  it('puts the chat’s readouts on the bar, workspace first, then plugins’, without waiting or hidden ones', () => {
    expect(ids(statusBarItems(BAR, PLUGINS))).toEqual([
      'workspace-cwd',
      'cache-hit-rate',
      'approval-mode',
      'some-plugin:item'
    ])
  })

  it('keeps everything else in the System popover, minus what the chrome already carries', () => {
    expect(ids(systemStatusItems(BAR, PLUGINS, true))).toEqual(['agents', 'version-client'])
  })

  it('returns the bar’s items to the popover while it is collapsed', () => {
    expect(ids(systemStatusItems(BAR, PLUGINS, false))).toEqual([
      'workspace-cwd',
      'agents',
      'context-usage',
      'cache-hit-rate',
      'session-timer',
      'approval-mode',
      'version-client',
      'some-plugin:item'
    ])
  })
})
