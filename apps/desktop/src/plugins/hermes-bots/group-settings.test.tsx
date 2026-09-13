import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createGroupGateway, scriptedStorage } from './group-test-utils'
import { translateBots } from './i18n-test-helper'
import type { GroupChat } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')
  const base = await pluginSdkMock(host)
  const dialog = await import('@/components/ui/dialog')
  const { Button } = await import('@/components/ui/button')
  const { Input } = await import('@/components/ui/input')
  const { useStore } = await import('@nanostores/react')

  return {
    ...base,
    ...dialog,
    Button,
    Input,
    useValue: useStore,
    useI18n: () => ({ t: { common: { cancel: 'Cancel', save: 'Save' } } }),
    usePluginI18n: () => translateBots
  }
})
vi.mock('./group-chat-parts', () => ({
  GroupClarifyCard: () => null,
  GroupImageControls: () => null,
  GroupMentionInput: () => null
}))

afterEach(cleanup)

async function setup() {
  vi.resetModules()
  const gateway = createGroupGateway()
  Object.assign(host, gateway.host)

  const [chat, view, shared] = await Promise.all([
    import('./group-chat'),
    import('./group-chat-view'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))
  chat.$groupChats.set({
    Count: { roomId: 'count', log: [], watermarks: {}, maxBotTurns: 4 },
    Other: { log: [], watermarks: {}, maxBotTurns: 2 }
  })
  const onClose = vi.fn()
  const onRenamed = vi.fn()
  const component = render(<view.GroupChatSettingsDialog group="Count" onClose={onClose} onRenamed={onRenamed} open />)

  return { ...component, chat, view, gateway, onClose, onRenamed }
}

describe('group settings reply budget', () => {
  it('saves through rename, persists, and reopens the new group with its budget', async () => {
    const room = await setup()
    const input = screen.getByRole('spinbutton', { name: 'Max bot turns' }) as HTMLInputElement
    expect(input.value).toBe('4')
    expect(input.min).toBe('1')
    expect(input.max).toBe('100')
    expect(screen.getByText(/Total bot replies across this group per user send/)).toBeTruthy()
    fireEvent.change(input, { target: { value: '20' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'Group name' }), { target: { value: 'Renamed' } })
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Save' })))
    expect(room.onClose).toHaveBeenCalledOnce()
    expect(room.onRenamed).toHaveBeenCalledWith('Renamed')
    expect(room.chat.$groupChats.get().Count).toBeUndefined()
    expect(room.chat.$groupChats.get().Renamed.maxBotTurns).toBe(20)
    const stored = JSON.parse(JSON.stringify(room.gateway.storage.get('group-chats'))) as Record<string, GroupChat>
    expect(stored.Renamed.maxBotTurns).toBe(20)
    expect(stored.Other.maxBotTurns).toBe(2)
    room.unmount()
    room.chat.$groupChats.set(stored)
    render(<room.view.GroupChatSettingsDialog group="Renamed" onClose={room.onClose} open />)
    expect((screen.getByRole('spinbutton') as HTMLInputElement).value).toBe('20')
  })

  it.each(['', '0', '101', '1.5'])('rejects invalid input %s without mutating even on submit', async value => {
    const room = await setup()
    const input = screen.getByRole('spinbutton')
    fireEvent.change(input, { target: { value } })
    expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true)
    expect(input.getAttribute('aria-invalid')).toBe('true')
    await act(async () => fireEvent.submit(input.closest('form')!))
    expect(room.chat.$groupChats.get().Count.maxBotTurns).toBe(4)
    expect(room.onClose).not.toHaveBeenCalled()
  })

  it('cancel discards edits and reopening restores saved values', async () => {
    const room = await setup()
    fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '99' } })
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(room.chat.$groupChats.get().Count.maxBotTurns).toBe(4)
    room.rerender(<room.view.GroupChatSettingsDialog group="Count" onClose={room.onClose} open={false} />)
    room.rerender(<room.view.GroupChatSettingsDialog group="Count" onClose={room.onClose} open />)
    expect((screen.getByRole('spinbutton') as HTMLInputElement).value).toBe('4')
  })
})
