/**
 * The New Group Chat picker rows are `label` flex containers wrapping a
 * `min-w-0 flex-1` text column whose lines are `truncate`. A flex/grid item
 * defaults to `min-width: auto`, so without `min-w-0` on the label itself the
 * row can never shrink below its longest line (`@handle · in "A", "B", …`),
 * `truncate` never engages, and the list widens past the viewport. The first
 * click then focuses a checkbox that sits off-screen and scrolls every name
 * out of view (PR #90624: wrapper 593px in a 414px viewport, scrollLeft 0 →
 * 178.38 on click; 414px / 0 → 0 with the class).
 *
 * jsdom does no layout, so this asserts the contract the layout engine needs:
 * the row label must opt out of the auto minimum width.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()

  return {
    ...original,
    host: {
      ...original.host,
      connections: vi.fn(async () => []),
      notify: vi.fn(),
      notifyError: vi.fn(),
      request: vi.fn(async () => ({})),
      requestProfile: vi.fn(async () => ({}))
    },
    // The plugin bundle normally lands via `ctx.i18n.register` at load.
    usePluginI18n: () => translateBots
  }
})

vi.setConfig({ testTimeout: 30_000 })

const LONG_NAME = 'An Extraordinarily Long Bot Name That Would Widen The Picker Row Past The Dialog Edge'

const roster: RosterRow[] = [
  { connectionId: 'local', display_name: LONG_NAME, name: 'long-bot' },
  { connectionId: 'local', name: 'short' }
]

const sevenBots: RosterRow[] = Array.from({ length: 7 }, (_, index) => ({
  connectionId: 'local',
  name: `bot-${index + 1}`
}))

beforeAll(async () => {
  // Radix Dialog reaches for APIs jsdom does not implement.
  Element.prototype.scrollIntoView = () => undefined
  Element.prototype.hasPointerCapture = () => false
  Element.prototype.releasePointerCapture = () => undefined
  Element.prototype.setPointerCapture = () => undefined
  await import('./create-dialog')
}, 120_000)

afterEach(() => {
  cleanup()
})

describe('CreateGroupChatDialog picker rows', () => {
  it('lets the row label shrink so long bot names truncate instead of widening the list', async () => {
    const { CreateGroupChatDialog } = await import('./create-dialog')

    render(<CreateGroupChatDialog onClose={() => undefined} onCreated={() => undefined} open roster={roster} />)

    const nameCell = screen.getByText(LONG_NAME)
    const label = nameCell.closest('label')

    expect(label).not.toBeNull()
    // The row is a flex container AND a grid item; both default to
    // `min-width: auto`, which is what defeats the inner `truncate`.
    expect(label!.className.split(/\s+/)).toContain('min-w-0')
    expect(label!.className.split(/\s+/)).toContain('flex')
    // The text column it wraps is the one that actually ellipsizes.
    expect(nameCell.className.split(/\s+/)).toContain('truncate')
    expect(nameCell.parentElement!.className.split(/\s+/)).toEqual(expect.arrayContaining(['min-w-0', 'flex-1']))

    // Every row gets the same treatment, not just the long one.
    const labels = screen.getAllByRole('checkbox').map(box => box.closest('label'))

    expect(labels).toHaveLength(roster.length)

    for (const row of labels) {
      expect(row!.className.split(/\s+/)).toContain('min-w-0')
    }
  })

  it('allows a seventh bot only after the room limit is raised', async () => {
    const { CreateGroupChatDialog } = await import('./create-dialog')
    const onCreated = vi.fn()

    render(<CreateGroupChatDialog onClose={() => undefined} onCreated={onCreated} open roster={sevenBots} />)

    const boxes = screen.getAllByRole('checkbox') as HTMLButtonElement[]

    for (const box of boxes.slice(0, 6)) {
      fireEvent.click(box)
    }

    expect(screen.getByText(/Pick 2–6 bots/)).toBeTruthy()
    expect(boxes[6].disabled).toBe(true)

    fireEvent.change(screen.getByLabelText('Maximum bots'), { target: { value: '7' } })
    expect(screen.getByText(/Pick 2–7 bots/)).toBeTruthy()
    expect(boxes[6].disabled).toBe(false)
    fireEvent.click(boxes[6])
    fireEvent.click(screen.getByRole('button', { name: 'Create Group (7)' }))

    const chat = await import('./group-chat')
    const created = Object.values(chat.$groupChats.get()).at(-1)

    expect(onCreated).toHaveBeenCalledTimes(1)
    expect(created?.members).toHaveLength(7)
    expect(created?.memberLimit).toBe(7)
  })
})
