// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { ProfileScope } from '@/api/client'
import { $terminalFontFamily } from '@/app/right-sidebar/terminal/terminal-font'
import type * as HermesModule from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { $settingsOwner, setSettingsScope } from '@/store/settings-scope'
import { $chatFontFamily } from '@/themes/chat-font'

import { hermesConfigKey } from '../hooks/use-config-record'

import { ResumeLastSessionSetting } from './appearance-settings'
import { ChatFontSetting } from './chat-font-setting'
import { TerminalFontSetting } from './terminal-font-setting'

const mocks = vi.hoisted(() => ({ save: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesModule>()),
  getHermesConfigRecord: vi.fn(async (owner: ProfileScope) => {
    const isGatewayB = typeof owner === 'object' && owner?.connectionOwner?.token === 'synthetic-b'
    const font = isGatewayB ? 'B Font' : 'A Font'

    return {
      desktop: { font_family: font },
      display: { resume_last_session: !isGatewayB },
      terminal: { font_family: font }
    }
  }),
  saveHermesConfig: (...args: unknown[]) => mocks.save(...args)
}))
vi.mock('@/store/notifications', () => ({ notifyError: (...args: unknown[]) => mocks.notifyError(...args) }))

const gatewayA = {
  baseUrl: 'https://gateway-a.example',
  connectionId: 'gateway',
  mode: 'remote',
  profile: 'default',
  token: 'synthetic-a'
} as const

const cases = [
  { Component: ChatFontSetting, name: 'Chat Font', section: 'desktop', font: $chatFontFamily },
  { Component: TerminalFontSetting, name: 'Terminal Font', section: 'terminal', font: $terminalFontFamily }
]

async function settle() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1)
  })
}

async function debounce() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(550)
  })
}

beforeEach(() => {
  vi.useFakeTimers()
  queryClient.clear()
  mocks.save.mockReset().mockResolvedValue({ ok: true })
  mocks.notifyError.mockClear()
  $activeGatewayProfile.set('default')
  setSettingsScope('default')
  $connection.set(gatewayA as never)
})
afterEach(() => {
  cleanup()
  $connection.set(null)
  queryClient.clear()
  vi.useRealTimers()
})

it.each(cases)(
  '$name drops unsent drafts on real same-profile owner changes, including same-id replacement',
  async ({ Component, name, font }) => {
    render(
      <QueryClientProvider client={queryClient}>
        <Component />
      </QueryClientProvider>
    )
    await settle()

    for (const connectionId of ['other-gateway', 'gateway']) {
      fireEvent.change(screen.getByRole('combobox', { name }), { target: { value: 'A Draft' } })
      act(() => $connection.set({ ...gatewayA, connectionId, token: 'synthetic-b' } as never))
      await settle()
      await debounce()
      expect((screen.getByRole('combobox', { name }) as HTMLInputElement).value).toBe('B Font')
      expect(font.get()).toBe('B Font')
      expect(mocks.save).not.toHaveBeenCalled()
      act(() => $connection.set(gatewayA as never))
      await settle()
      expect(font.get()).toBe('A Font')
    }
  }
)

it.each(cases)(
  '$name ignores delayed success/failure from the previous owner even after B edits',
  async ({ Component, name, section, font }) => {
    render(
      <QueryClientProvider client={queryClient}>
        <Component />
      </QueryClientProvider>
    )
    await settle()

    for (const [ok, editB] of [
      [true, false],
      [false, false],
      [true, true],
      [false, true]
    ]) {
      let complete!: (result: { ok: boolean }) => void
      mocks.save.mockImplementationOnce(
        () =>
          new Promise(resolve => {
            complete = resolve
          })
      )
      const ownerA = $settingsOwner.get()
      fireEvent.change(screen.getByRole('combobox', { name }), { target: { value: 'A Draft' } })
      await debounce()
      expect(mocks.save).toHaveBeenLastCalledWith({ [section]: { font_family: 'A Draft' } }, ownerA)
      act(() => $connection.set({ ...gatewayA, token: 'synthetic-b' } as never))
      await settle()
      const ownerB = $settingsOwner.get()

      if (editB) {
        fireEvent.change(screen.getByRole('combobox', { name }), { target: { value: 'B Draft' } })
      }

      await act(async () => {
        complete({ ok })
        await Promise.resolve()
      })
      expect(font.get()).toBe(editB ? 'B Draft' : 'B Font')
      expect(mocks.notifyError).not.toHaveBeenCalled()
      expect(queryClient.getQueryData(hermesConfigKey(ownerA!))).toMatchObject({ [section]: { font_family: 'A Font' } })
      expect(queryClient.getQueryData(hermesConfigKey(ownerB!))).toMatchObject({ [section]: { font_family: 'B Font' } })
      await debounce()

      if (editB) {
        expect(mocks.save).toHaveBeenLastCalledWith({ [section]: { font_family: 'B Draft' } }, ownerB)
      } else {
        expect(mocks.save).toHaveBeenLastCalledWith({ [section]: { font_family: 'A Draft' } }, ownerA)
      }

      act(() => $connection.set(gatewayA as never))
      await settle()
    }
  }
)

it('Resume last session reads and writes the same exact Settings owner as the font controls', async () => {
  render(
    <QueryClientProvider client={queryClient}>
      <ResumeLastSessionSetting />
    </QueryClientProvider>
  )
  await settle()

  const ownerA = $settingsOwner.get()
  fireEvent.click(screen.getByRole('switch'))
  expect(mocks.save).toHaveBeenLastCalledWith({ display: { resume_last_session: false } }, ownerA)

  act(() => $connection.set({ ...gatewayA, token: 'synthetic-b' } as never))
  await settle()
  const ownerB = $settingsOwner.get()
  fireEvent.click(screen.getByRole('switch'))
  expect(mocks.save).toHaveBeenLastCalledWith({ display: { resume_last_session: true } }, ownerB)
})
