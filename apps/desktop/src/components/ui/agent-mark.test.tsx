import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { BotBadge } from '@/plugins/hermes-bots/avatar'
import { $profileColors } from '@/store/profile'
import { $shellChoice } from '@/store/shell-variant'

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
  $profileColors.set({})
})

const badge = () => (
  <div data-testid="badge">
    <BotBadge color="#cc8855" name="research" shape="drop" size={24} />
  </div>
)

it('draws a bot as the redesign’s agent tile, in that agent’s colour and at the asked size', () => {
  $shellChoice.set(true)
  $profileColors.set({ research: '#00f279' })
  render(badge())

  const tile = screen.getByTestId('badge').querySelector('[data-bot-face="research"] > span') as HTMLElement

  expect(tile.querySelector('svg[data-bot-face], svg[data-hb-math]')).toBeNull()
  expect(tile.querySelector('img')).not.toBeNull()
  expect(tile.style.width).toBe('24px')
  expect(tile.style.backgroundColor).toBe('rgb(14, 154, 107)')
})

it('keeps the bot’s own face in classic', () => {
  $shellChoice.set(false)
  render(badge())

  expect(screen.getByTestId('badge').querySelector('svg')).not.toBeNull()
  expect(screen.getByTestId('badge').querySelector('img')).toBeNull()
  // One marker per bot: a row is looked up by it.
  expect(screen.getByTestId('badge').querySelectorAll('[data-bot-face="research"]')).toHaveLength(1)
})
