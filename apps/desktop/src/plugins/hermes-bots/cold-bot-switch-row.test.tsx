/**
 * The pending cold-open mark is a spinner on the clicked row, not a new owner.
 * Highlight still follows the chat on screen (hermes-agent#120277).
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { BotRow } from './bot-row'
import { $botChatFocused, $pendingBotOpen, $selectedRosterKey } from './bot-state'
import { $groupChatWorkspace } from './group-chat'
import { translateBotsIn } from './i18n-test-helper'
import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    usePluginI18n: () => translateBotsIn('en')
  }
})

const noop = () => undefined

const alpha = { connectionId: 'local', name: 'alpha' } as RosterRow
const bravo = { connectionId: 'local', name: 'bravo' } as RosterRow

beforeEach(() => {
  $groupChatWorkspace.set(null)
  $botChatFocused.set(false)
  $selectedRosterKey.set('local::alpha')
  $pendingBotOpen.set({ generation: 1, key: 'local::bravo' })
})

afterEach(() => {
  $pendingBotOpen.set(null)
  $selectedRosterKey.set('')
  cleanup()
})

it('spins the pending target row without stealing the highlight', () => {
  render(
    <>
      <BotRow bot={alpha} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />
      <BotRow bot={bravo} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />
    </>
  )

  const [alphaRow, bravoRow] = screen.getAllByRole('button')

  expect(bravoRow.getAttribute('aria-busy')).toBe('true')
  expect(bravoRow.querySelector('[role="status"]')?.getAttribute('aria-label')).toBe('Opening chat…')
  expect(alphaRow.getAttribute('aria-busy')).not.toBe('true')
  expect(alphaRow.querySelector('[role="status"]')).toBeNull()
  expect(alphaRow.className).toContain('bg-(--ui-row-active-background)')
  expect(bravoRow.className).not.toContain('bg-(--ui-row-active-background)')
})
