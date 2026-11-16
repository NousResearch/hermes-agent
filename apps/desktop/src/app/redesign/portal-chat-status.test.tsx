import { describe, expect, it } from 'vitest'

import { STATUSBAR_WAITING, type StatusbarItem } from '@/app/shell/statusbar-controls'

import { chatStatusItems, systemStatusItems } from './portal-chat-status'

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

const ids = (items: readonly StatusbarItem[]) => items.map(entry => entry.id)

describe('the redesign’s split of the statusbar', () => {
  it('puts the chat’s own readouts on its line, workspace first, without waiting or hidden ones', () => {
    expect(ids(chatStatusItems(BAR))).toEqual(['workspace-cwd', 'cache-hit-rate', 'approval-mode'])
  })

  it('keeps everything else in the System popover, minus what the chrome already carries', () => {
    expect(ids(systemStatusItems(BAR, true))).toEqual(['agents', 'version-client', 'some-plugin:item'])
  })

  it('returns the chat’s readouts to the popover while its line is off', () => {
    expect(ids(systemStatusItems(BAR, false))).toEqual([
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
