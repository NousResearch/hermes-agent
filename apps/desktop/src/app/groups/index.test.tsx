import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $profiles } from '@/store/profile'
import { $projects } from '@/store/projects'
import type { ProfileInfo } from '@/types/hermes'

import { GroupsView } from '.'

const profile = (name: string, displayName?: string): ProfileInfo => ({
  display_name: displayName,
  has_env: true,
  is_default: name === 'default',
  model: null,
  name,
  path: `/profiles/${name}`,
  provider: null,
  skill_count: 0
})

describe('GroupsView', () => {
  beforeEach(() => {
    $projects.set([])
    HTMLElement.prototype.scrollIntoView = vi.fn()
  })

  it('does not require a view-local gateway event listener', async () => {
    render(<GroupsView navigate={vi.fn()} request={vi.fn(async () => ({ rooms: [] }))} roomId={null} />)
    await screen.findByText('No group rooms yet.')
  })

  it('selects existing profiles by display name and sends canonical names', async () => {
    $profiles.set([
      profile('default', 'Hermes'),
      profile('planner', 'Launch Planner'),
      profile('reviewer', 'Risk Reviewer')
    ])

    const request = vi.fn(async (method: string) => method === 'group.room.list'
      ? { rooms: [] }
      : method === 'group.room.create'
        ? { room: { id: 'r-picker', name: 'Launch', profiles: ['planner'], messages: [] } }
        : { ok: true })

    render(<GroupsView navigate={vi.fn()} request={request} roomId={null} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Create room' }))
    fireEvent.change(screen.getByLabelText('Room name'), { target: { value: 'Launch' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'Search profiles' }), { target: { value: 'launch' } })
    fireEvent.click(screen.getByRole('checkbox', { name: /Launch Planner.*planner/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Create' }))

    await waitFor(() => expect(request).toHaveBeenCalledWith('group.room.create', {
      name: 'Launch', profiles: ['planner']
    }))
  })

  it('selects a project workspace for creation and shows it in room details', async () => {
    $profiles.set([profile('planner')])
    $projects.set([{
      id: 'p_launch', slug: 'launch', name: 'Launch project', description: null, icon: null, color: null,
      board_slug: null, primary_path: '/work/launch', archived: false, created_at: 1,
      folders: [{ path: '/work/launch', label: null, is_primary: true, added_at: 1 }]
    }])

    const request = vi.fn(async (method: string) => method === 'group.room.list'
      ? { rooms: [] }
      : method === 'group.room.create'
        ? { room: { id: 'r-work', name: 'Launch', profiles: ['planner'], workspace: '/work/launch', messages: [] } }
        : { ok: true })

    const { unmount } = render(<GroupsView navigate={vi.fn()} request={request} roomId={null} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Create room' }))
    fireEvent.change(screen.getByLabelText('Room name'), { target: { value: 'Launch' } })
    fireEvent.click(screen.getByRole('checkbox', { name: 'planner (planner)' }))
    fireEvent.click(screen.getByLabelText('Workspace'))
    fireEvent.click(await screen.findByRole('option', { name: /Launch project.*\/work\/launch/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => expect(request).toHaveBeenCalledWith('group.room.create', {
      name: 'Launch', profiles: ['planner'], workspace: '/work/launch'
    }))
    unmount()

    render(<GroupsView navigate={vi.fn()} request={async method => method === 'group.room.get'
      ? { room: { id: 'r-work', name: 'Launch', profiles: ['planner'], workspace: '/work/launch', messages: [] } }
      : { ok: true }} roomId="r-work" />)
    expect(await screen.findByText(/Workspace:\s*\/work\/launch/)).toBeTruthy()
  })

  it('loads older history with cursor pagination and shows compression context', async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method !== 'group.room.get') {return { ok: true }}

      if (params?.cursor === 'older-1') {
        return { room: { id: 'r-page', name: 'Paged', profiles: [], messages: [
          { id: 'm1', seq: 1, role: 'user', content: 'Oldest' },
          { id: 'm2', seq: 2, role: 'assistant', content: 'Boundary' }
        ] }, has_more: false }
      }

      return { room: {
        id: 'r-page', name: 'Paged', profiles: [], context_status: 'compressed', summary: 'Earlier context summary',
        messages: [{ id: 'm2', seq: 2, role: 'assistant', content: 'Boundary' }, { id: 'm3', seq: 3, role: 'assistant', content: 'Latest' }]
      }, cursor: 'older-1', has_more: true }
    })

    render(<GroupsView navigate={vi.fn()} request={request} roomId="r-page" />)
    expect(await screen.findByText('Earlier context summary')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Load earlier messages' }))
    expect(await screen.findByText('Oldest')).toBeTruthy()
    expect(screen.getAllByText('Boundary')).toHaveLength(1)
    expect(request).toHaveBeenCalledWith('group.room.get', { room_id: 'r-page', cursor: 'older-1', before_seq: 2 })
  })

  it('creates a multi-profile room, sends @all, and stops the room', async () => {
    $profiles.set([profile('planner'), profile('reviewer')])

    const request = vi.fn(async (method: string) => {
      if (method === 'group.room.list') {return { rooms: [] }}

      if (method === 'group.room.create') {return { room: { id: 'r1', name: 'Launch', profiles: ['planner', 'reviewer'], messages: [] } }}

      return { ok: true }
    })

    const navigate = vi.fn()
    render(<GroupsView navigate={navigate} request={request} roomId={null} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Create room' }))
    fireEvent.change(screen.getByLabelText('Room name'), { target: { value: 'Launch' } })
    fireEvent.click(screen.getByRole('checkbox', { name: 'planner (planner)' }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'reviewer (reviewer)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => expect(navigate).toHaveBeenCalledWith('/groups/r1'))

    render(<GroupsView navigate={navigate} request={request} roomId="r1" />)
    const composer = await screen.findByLabelText('Message the room')
    fireEvent.change(composer, { target: { value: '@all ship it' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    await waitFor(() => expect(request).toHaveBeenCalledWith('group.message.send', {
      room_id: 'r1', content: '@all ship it', mentions: ['all']
    }))
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
    await waitFor(() => expect(request).toHaveBeenCalledWith('group.run.interrupt', { room_id: 'r1' }))
  })
})
