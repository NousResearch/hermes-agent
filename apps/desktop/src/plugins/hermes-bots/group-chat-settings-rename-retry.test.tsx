/**
 * The rename-then-failed-members path (review follow-up on #91389).
 *
 * Save does two durable things in sequence: it renames the room, then replaces
 * its member set. If the rename lands and the member write does not, the room
 * no longer answers to the name the dialog opened with — so a naive retry
 * would call renameGroupChat against a key that no longer exists and the user
 * could never recover through the UI, only by closing and reopening.
 *
 * The dialog therefore tracks the last name that actually took, and bases a
 * retry on that. This is the path most likely to corrupt durable state, so it
 * is asserted end to end through the real dialog rather than on the helper.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupMembership from './group-membership'
import type { GroupChat, GroupMember } from './types'

// Radix calls these on open; jsdom does not implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const notifyError = vi.fn()
const notify = vi.fn()

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, host: { ...sdk.host, notify, notifyError, request: vi.fn(async () => ({})) } }
})

// Fail every member replacement, so Save always gets past the rename and then
// stops on the members step — the exact ordering the guard exists for.
const replaceGroupChatMembers = vi.fn(async () => {
  throw new Error('gateway refused the member write')
})

vi.mock('./group-membership', async importOriginal => {
  const actual = await importOriginal<typeof groupMembership>()

  return { ...actual, replaceGroupChatMembers }
})

const { $groupChats } = await import('./group-chat')
const { $botMeta, $lastRoster } = await import('./data')
const { GroupChatSettingsDialog } = await import('./group-chat-view')

const MEMBERS: GroupMember[] = [{ name: 'pm' }, { name: 'scout' }] as GroupMember[]

function room(name: string): Record<string, GroupChat> {
  return {
    [name]: { log: [], members: MEMBERS, roomId: 'room-1', watermarks: {} } as unknown as GroupChat
  }
}

beforeEach(() => {
  notify.mockClear()
  notifyError.mockClear()
  replaceGroupChatMembers.mockClear()
  $groupChats.set(room('Ops'))
  $botMeta.set({ pm: { groups: ['Ops'] }, scout: { groups: ['Ops'] } })
  $lastRoster.set(MEMBERS as never)
})

afterEach(cleanup)

// Two i18n sources render differently under test: the CORE table (useI18n →
// t.common.save) resolves to real strings, while the plugin's own table
// (useBots → b.group.*) is not loaded and echoes its keys back. Queries below
// match each accordingly rather than assuming one convention.
const nameInput = () => screen.getByLabelText('group.nameLabel')
const saveButton = () => screen.getByRole('button', { name: 'Save' })

async function saveOnce() {
  const before = notifyError.mock.calls.length

  fireEvent.click(saveButton())
  await waitFor(() => expect(notifyError.mock.calls.length).toBeGreaterThan(before))
}

describe('saving a rename whose member write then fails', () => {
  it('keeps the renamed room and retries against the NEW name, not the one the dialog opened with', async () => {
    render(<GroupChatSettingsDialog group="Ops" members={MEMBERS} onClose={vi.fn()} open />)

    fireEvent.change(nameInput(), { target: { value: 'Operations' } })
    await saveOnce()

    // The rename landed even though the member write did not.
    expect(Object.keys($groupChats.get())).toContain('Operations')
    expect(replaceGroupChatMembers).toHaveBeenCalledWith('Operations', expect.anything())

    // Retry. The room key 'Ops' is gone, so a retry that re-ran the rename
    // against it would target a room that no longer exists.
    replaceGroupChatMembers.mockClear()
    await saveOnce()

    expect(replaceGroupChatMembers).toHaveBeenCalledWith('Operations', expect.anything())
    expect(Object.keys($groupChats.get())).toContain('Operations')
    expect(Object.keys($groupChats.get())).not.toContain('Ops')
  })

  it('leaves the dialog open on the failure so the user can retry at all', async () => {
    const onClose = vi.fn()
    render(<GroupChatSettingsDialog group="Ops" members={MEMBERS} onClose={onClose} open />)

    fireEvent.change(nameInput(), { target: { value: 'Operations' } })
    await saveOnce()

    expect(onClose).not.toHaveBeenCalled()
  })

  it('does not report a rename as renamed while the save has not completed', async () => {
    const onRenamed = vi.fn()
    render(<GroupChatSettingsDialog group="Ops" members={MEMBERS} onClose={vi.fn()} onRenamed={onRenamed} open />)

    fireEvent.change(nameInput(), { target: { value: 'Operations' } })
    await saveOnce()

    // onRenamed drives the caller's workspace key. Firing it before the save
    // finished would point the open room at a half-applied edit.
    expect(onRenamed).not.toHaveBeenCalled()
  })
})
