import type * as HermesSdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider, useI18n } from '@/i18n'
import type { I18nContextValue } from '@/i18n'
import { registerPluginLocales } from '@/i18n/plugin-i18n'
import { BOTS_LOCALES } from '@/plugins/hermes-bots/i18n'
import { translateBotsIn } from '@/plugins/hermes-bots/i18n-test-helper'
import { McpSetupButton } from '@/plugins/hermes-bots/mcp-setup'
import { AdvancedProfileConfig, applyAdvancedConfig, emptyAdvancedState } from '@/plugins/hermes-bots/profile-config'

const mocks = vi.hoisted(() => ({
  request: vi.fn(async (method: string, _params?: Record<string, unknown>): Promise<unknown> => {
    if (method === 'profiles.describe') {
      return {
        soul: 'My English persona',
        model: { provider: 'my-provider', default: 'my-model' },
        skills: [{ name: 'my-skill', enabled: true }],
        toolsets: [{ name: 'my-tools', enabled: true }],
        mcp_servers: [{ name: 'my-mcp', enabled: true }]
      }
    }

    if (method === 'skills.manage') {
      return { results: [{ name: 'user-skill', description: 'English description from the server' }] }
    }

    if (method === 'profiles.configure') {
      return { applied: { skills: true } }
    }

    return {}
  }),
  notify: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    CapabilitiesView: undefined,
    ConnectorsTab: undefined,
    ToolsetConfigPanel: undefined,
    host: { ...sdk.host, request: mocks.request, notify: mocks.notify }
  }
})
let i18n: I18nContextValue
let dispose: () => void

function Controls() {
  i18n = useI18n()

  return null
}

function mount(children: React.ReactNode) {
  dispose = registerPluginLocales('hermes-bots', BOTS_LOCALES)

  return render(
    <I18nProvider configClient={null} initialLocale="zh">
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <Controls />
        {children}
      </QueryClientProvider>
    </I18nProvider>
  )
}

const bot = { name: 'writer' }

function Editor() {
  const [state, setState] = useState(emptyAdvancedState)

  return (
    <>
      <AdvancedProfileConfig bot={bot} setState={setState} state={state} />
      <button onClick={() => void applyAdvancedConfig(bot, state)}>Persist fixture</button>
    </>
  )
}

afterEach(() => {
  cleanup()
  dispose()
  vi.clearAllMocks()
})
it('localizes the advanced editor and hub without translating capability IDs or user data', async () => {
  mount(<Editor />)
  const zh = translateBotsIn('zh')
  expect(await screen.findByText(zh('editor.editSoul'))).toBeTruthy()
  expect(screen.getByText(zh('editor.skillsEnabled', 1, 1))).toBeTruthy()
  expect(await screen.findByText(i18n.t.settings.model.provider)).toBeTruthy()
  fireEvent.click(screen.getByRole('checkbox', { name: 'my-skill' }))
  fireEvent.change(screen.getByPlaceholderText(zh('tools.searchHub')), { target: { value: 'my query' } })
  await act(() => i18n.setLocale('ja'))
  const ja = translateBotsIn('ja')
  expect(screen.getByText(ja('editor.skillsEnabled', 0, 1))).toBeTruthy()

  for (const value of ['My English persona', 'my-provider', 'my-model', 'my query']) {
    expect(screen.getByDisplayValue(value)).toBeTruthy()
  }

  fireEvent.click(screen.getByRole('button', { name: i18n.t.skills.hub.search }))
  expect(await screen.findByText('English description from the server')).toBeTruthy()
  expect(screen.getByRole('button', { name: ja('tools.installHint', 'user-skill') })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Persist fixture' }))
  await waitFor(() =>
    expect(mocks.request).toHaveBeenCalledWith('profiles.configure', { name: 'writer', disabled_skills: ['my-skill'] })
  )
})
it('localizes MCP setup actions while retaining the profile, preset and environment identifiers', async () => {
  mount(
    <McpSetupButton entry={{ name: 'my-server', requires: ['MY_SERVER_TOKEN'], fromCatalog: true }} profile="writer" />
  )
  const zh = translateBotsIn('zh')
  fireEvent.click(await screen.findByRole('button', { name: zh('tools.setUp') }))
  expect(await screen.findByPlaceholderText('MY_SERVER_TOKEN')).toBeTruthy()
  await act(() => i18n.setLocale('zh-hant'))
  expect(screen.getByRole('button', { name: translateBotsIn('zh-hant')('tools.saveTest') })).toBeTruthy()
  expect(mocks.request).toHaveBeenCalledWith('mcp.servers.add', {
    profile: 'writer',
    name: 'my-server',
    preset: 'my-server'
  })
  expect(mocks.request.mock.calls.some(([method]) => method === 'mcp.servers.set_api_key')).toBe(false)
})
