// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import type { DiscordBotCheckResponse, MessagingPlatformInfo } from '@/hermes'

import { DiscordBotCheck } from './discord-bot-check'

const checkDiscordBot = vi.fn()
const openExternalLink = vi.fn()

vi.mock('@/hermes', () => ({ checkDiscordBot: (profile?: string) => checkDiscordBot(profile) }))
vi.mock('@/lib/external-link', () => ({ openExternalLink: (url: string) => openExternalLink(url) }))

afterEach(() => {
  cleanup()
  checkDiscordBot.mockReset()
  openExternalLink.mockReset()
})

const discord = (tokenSet: boolean) =>
  ({
    id: 'discord',
    name: 'Discord',
    env_vars: [{ key: 'DISCORD_BOT_TOKEN', is_set: tokenSet }]
  }) as MessagingPlatformInfo

const bot: DiscordBotCheckResponse = {
  app_id: '42',
  bot_name: 'Hermes Bot',
  bot_settings_url: 'https://discord.com/developers/applications/42/bot',
  invite_url: 'https://discord.com/oauth2/authorize?client_id=42',
  message_content: false,
  owners: [],
  server_count: 0,
  server_members: false
}

test('the check waits for a saved token', () => {
  render(<DiscordBotCheck platform={discord(false)} scopeProfile="work" />)

  expect((screen.getByRole('button', { name: 'Check bot' }) as HTMLButtonElement).disabled).toBe(true)
  expect(screen.getByText('Save a Discord bot token first.')).toBeTruthy()
})

test('a bot with Message Content Intent off links to its Bot settings and its invite', async () => {
  checkDiscordBot.mockResolvedValueOnce(bot)
  render(<DiscordBotCheck platform={discord(true)} scopeProfile="work" />)

  fireEvent.click(screen.getByRole('button', { name: 'Check bot' }))

  expect(await screen.findByText('Hermes Bot')).toBeTruthy()
  expect(checkDiscordBot).toHaveBeenCalledWith('work')
  expect(screen.getByText(/Message Content Intent is off/)).toBeTruthy()
  expect(screen.getByText("The bot isn't in any server yet.")).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Open Bot settings' }))
  fireEvent.click(screen.getByRole('button', { name: 'Add to a server' }))
  expect(openExternalLink.mock.calls).toEqual([[bot.bot_settings_url], [bot.invite_url]])
  expect(screen.getByRole('button', { name: 'Check again' })).toBeTruthy()
})

test('a refused check shows the backend sentence, not the IPC envelope', async () => {
  const detail = 'Discord rejected the saved token. Reset it on the Bot page and paste the new one.'
  checkDiscordBot.mockRejectedValueOnce(
    new Error(`Error invoking remote method 'hermes:api': Error: 400: ${JSON.stringify({ detail })}`)
  )
  render(<DiscordBotCheck platform={discord(true)} scopeProfile={undefined} />)

  fireEvent.click(screen.getByRole('button', { name: 'Check bot' }))

  expect(await screen.findByText(detail)).toBeTruthy()
  expect(screen.queryByText(/Error invoking remote method|"detail"/)).toBeNull()
})
