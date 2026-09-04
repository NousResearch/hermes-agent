import type * as HermesSdk from '@hermes/plugin-sdk'
import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'

import { GroupRow } from './bot-row'
import { $groupChats } from './group-chat'
import { GroupChatSettingsDialog } from './group-chat-view'
import { translateBots } from './i18n-test-helper'

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
    onSettings
  }

  render(<GroupRow {...props} />)
  fireEvent.contextMenu(screen.getByRole('button'))
  fireEvent.click(await screen.findByText('Group settings'))
  expect(onSettings).toHaveBeenCalledWith({ members: [], name: 'Planning' })
})

it('blocks only a changed name while a reply is active', () => {
  $groupChats.set({ Old: { log: [], watermarks: {}, running: true } })
  render(<GroupChatSettingsDialog group="Old" members={[]} onClose={vi.fn()} open />)
  const save = screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement
  const input = screen.getByRole('textbox', { name: 'Group name' })
  expect(save.disabled).toBe(false)
  fireEvent.change(input, { target: { value: 'New' } })
  expect(save.disabled).toBe(true)
  expect(screen.getByText('Wait for the current replies to finish before renaming this group chat.')).toBeTruthy()
  fireEvent.change(input, { target: { value: 'Old' } })
  expect(save.disabled).toBe(false)
})
