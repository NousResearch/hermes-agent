import { describe, expect, it } from 'vitest'

import type { Contribution } from '@/contrib/types'

import { group } from '../model'

import { rootChildSide } from './track-model'

type PaneData = {
  collapsible?: boolean
  placement?: string
  revealAliases?: string[]
  uncloseable?: boolean
}

const pane = (id: string, data: PaneData): Contribution => ({ area: 'panes', data, id })

const productionFiles: PaneData = {
  collapsible: true,
  placement: 'right',
  revealAliases: ['file-browser']
}

describe('rootChildSide', () => {
  it.each([
    {
      expected: 'right',
      name: 'keeps a Files edge anchor right when ordinary left and main panes share its group',
      panes: [
        pane('files', productionFiles),
        pane('contributed-left', { placement: 'left' }),
        pane('contributed-main', { placement: 'main' })
      ]
    },
    {
      expected: 'left',
      name: 'keeps the shipped Quad Sessions + Files group on the left',
      panes: [
        pane('sessions', { collapsible: true, placement: 'left', revealAliases: ['chat-sidebar'] }),
        pane('files', productionFiles)
      ]
    },
    {
      expected: null,
      name: 'keeps the Focus Workspace + Files group main-owned',
      panes: [
        pane('workspace', { placement: 'main', uncloseable: true }),
        pane('files', productionFiles),
        pane('review', { collapsible: true, placement: 'right' }),
        pane('terminal', { placement: 'bottom' })
      ]
    },
    {
      expected: 'right',
      name: 'keeps a bottom-only terminal group on its prior right side',
      panes: [pane('terminal', { placement: 'bottom' })]
    },
    {
      expected: null,
      name: 'leaves a closeable main-only session tile group unowned',
      panes: [pane('session-tile', { placement: 'main' })]
    }
  ])('$name', ({ expected, panes }) => {
    const byId = new Map(panes.map(contribution => [contribution.id, contribution]))

    expect(rootChildSide(group(panes.map(contribution => contribution.id)), id => byId.get(id))).toBe(expected)
  })
})
