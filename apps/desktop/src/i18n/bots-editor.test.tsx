import type * as HermesSdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, expect, it, vi } from 'vitest'

import { I18nProvider, useI18n } from '@/i18n'
import type { I18nContextValue } from '@/i18n'
import { registerPluginLocales } from '@/i18n/plugin-i18n'
import { $imagenAvailable } from '@/plugins/hermes-bots/avatar-image'
import { CreateAgentDialog } from '@/plugins/hermes-bots/create-dialog'
import { EditProfileDialog } from '@/plugins/hermes-bots/edit-profile-dialog'
import { BOTS_LOCALES } from '@/plugins/hermes-bots/i18n'
import { translateBotsIn } from '@/plugins/hermes-bots/i18n-test-helper'

const mocks = vi.hoisted(() => ({
  request: vi.fn(async (_method: string) => ({ skills: [], toolsets: [], mcp_servers: [] })),
  connections: vi.fn(async () => [])
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, host: { ...sdk.host, request: mocks.request, connections: mocks.connections } }
})
let i18n: I18nContextValue
let dispose: (() => void) | undefined

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

beforeAll(() => {
  Element.prototype.scrollIntoView = () => undefined
  $imagenAvailable.set(false)
})
afterEach(() => {
  cleanup()
  dispose?.()
})
it('keeps a new bot draft when the language changes and localizes advanced controls', async () => {
  mount(<CreateAgentDialog onClose={() => undefined} open roster={[{ connectionId: 'local', name: 'default' }]} />)
  const zh = translateBotsIn('zh')
  expect(screen.getByText(zh('editor.newDescription'))).toBeTruthy()
  fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'draft-test' } })
  fireEvent.click(screen.getByRole('button', { name: zh('bot.advanced') }))
  expect(screen.getByText(zh('editor.cloneFrom'))).toBeTruthy()
  expect(screen.getByText(zh('editor.shareKeysHint'))).toBeTruthy()
  await act(() => i18n.setLocale('ja'))
  const ja = translateBotsIn('ja')
  expect(screen.getByText(ja('editor.cloneFrom'))).toBeTruthy()
  expect(screen.getByDisplayValue('draft-test')).toBeTruthy()
  expect(screen.getByRole('button', { name: ja('editor.createBot') })).toBeTruthy()
  expect(mocks.request.mock.calls.some(([method]) => method === 'profiles.create')).toBe(false)
})
it('localizes Edit Profile and image controls without losing user content', async () => {
  mount(
    <EditProfileDialog
      bot={{ connectionId: 'local', name: 'fixture-bot', description: 'user authored description' }}
      onClose={() => undefined}
      open
    />
  )
  const zh = translateBotsIn('zh')
  expect(screen.getByText(zh('editor.title'))).toBeTruthy()
  expect(screen.getByText(zh('editor.description'))).toBeTruthy()
  expect(screen.getByText(zh('editor.editDescription', 'Fixture Bot', 'fixture-bot'))).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: zh('avatar.upload') }))
  expect(screen.getByRole('button', { name: zh('editor.chooseImage') })).toBeTruthy()
  await act(() => i18n.setLocale('zh-hant'))
  const hant = translateBotsIn('zh-hant')
  expect(screen.getByRole('button', { name: hant('editor.chooseImage') })).toBeTruthy()
  expect(screen.getByDisplayValue('user authored description')).toBeTruthy()
})
