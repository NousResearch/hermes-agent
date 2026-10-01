import type * as HermesSdk from '@hermes/plugin-sdk'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { expect, it, vi } from 'vitest'

import { GroupRow } from './bot-row'
import { $botMeta } from './data'
import { $groupChats } from './group-chat'
import { translateBots } from './i18n-test-helper'
import { renderRosterDialogs } from './roster-pane-dialogs'
import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, usePluginI18n: () => translateBots }
})

it('opens the existing Group settings from the roster context menu', async () => {
  const onSettings = vi.fn()

  const props = {
    active: false,
    group: 'Planning',
    members: [],
    needsYou: false,
    onDisband: vi.fn(),
    onOpen: vi.fn(),
    onSettings,
    onNewSection: vi.fn()
  }

  render(<GroupRow {...props} />)
  fireEvent.contextMenu(screen.getByRole('button'))
  fireEvent.click(await screen.findByText('Group settings'))
  expect(onSettings).toHaveBeenCalledWith({ members: [], name: 'Planning' })
})

it('renames membership current at Save, not the roster row snapshot from opening settings', async () => {
  const alpha = { name: 'alpha' } as RosterRow
  const beta = { name: 'beta' } as RosterRow
  $groupChats.set({ Old: { log: [], members: [], watermarks: {} } })
  $botMeta.set({ alpha: { groups: ['Old'] }, beta: { groups: [] } })

  let changeMembership = () => undefined

  function RosterSettings() {
    const [editingGroup, setEditingGroup] = useState<null | string>(null)
    const [roster, setRoster] = useState([alpha, beta])

    changeMembership = () => {
      $botMeta.set({ alpha: { groups: [] }, beta: { groups: ['Old'] } })
      setRoster([...roster])
    }

    return (
      <>
        <GroupRow
          active={false}
          group="Old"
          members={[alpha]}
          needsYou={false}
          onDisband={vi.fn()}
          onNewSection={vi.fn()}
          onOpen={vi.fn()}
          onSettings={row => setEditingGroup(row.name)}
        />
        {renderRosterDialogs({
          b: { group: { deleteAction: 'Delete', deleteTitle: 'Delete group' }, bot: { deleteTitle: 'Delete bot' } } as unknown as Parameters<typeof renderRosterDialogs>[0]['b'],
          t: { common: { delete: 'Delete' } } as unknown as Parameters<typeof renderRosterDialogs>[0]['t'],
          createOpen: false,
          setCreateOpen: vi.fn(),
          groupCreateOpen: false,
          setGroupCreateOpen: vi.fn(),
          editing: null,
          setEditing: vi.fn(),
          deleting: null,
          setDeleting: vi.fn(),
          deletingGroup: null,
          setDeletingGroup: vi.fn(),
          editingGroup,
          setEditingGroup,
          grouping: null,
          setGrouping: vi.fn(),
          sectionDialog: null,
          setSectionDialog: vi.fn(),
          roster,
          activeSourceRoster: roster,
          refetch: vi.fn(async () => ({})) as unknown as Parameters<typeof renderRosterDialogs>[0]['refetch']
        })}
      </>
    )
  }

  render(<RosterSettings />)
  fireEvent.contextMenu(screen.getByRole('button', { name: /Old/ }))
  fireEvent.click(await screen.findByText('Group settings'))
  changeMembership()
  fireEvent.change(screen.getByRole('textbox', { name: 'Group name' }), { target: { value: 'New' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))

  await waitFor(() => expect($groupChats.get().New).toBeTruthy())
  await waitFor(() => expect($botMeta.get().beta.groups).toEqual(['New']))
  expect($botMeta.get().alpha.groups).toEqual([])
})
