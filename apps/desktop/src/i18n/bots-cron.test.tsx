import type * as HermesSdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, expect, it, vi } from 'vitest'

import { I18nProvider, useI18n } from '@/i18n'
import type { I18nContextValue } from '@/i18n'
import { registerPluginLocales } from '@/i18n/plugin-i18n'
import { CreateRoutineDialog, RoutineDetailDialog } from '@/plugins/hermes-bots/cron'
import { BOTS_LOCALES } from '@/plugins/hermes-bots/i18n'
import { translateBotsIn } from '@/plugins/hermes-bots/i18n-test-helper'

const { request } = vi.hoisted(() => ({
  request: vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({}))
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, host: { ...sdk.host, request, notify: vi.fn() } }
})
// The bot desktop preview is outside the scheduling dialog under test.
vi.mock('@/plugins/hermes-bots/screen-hero', () => ({ ScreenHero: () => null }))
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

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})
afterEach(() => {
  cleanup()
  dispose()
  vi.clearAllMocks()
})
it('switches scheduling copy without changing a staged schedule or user instruction', async () => {
  mount(<CreateRoutineDialog bot="writer" onClose={() => undefined} open />)
  const zh = translateBotsIn('zh')
  expect(screen.getByText(zh('cron.stopAfter'))).toBeTruthy()
  fireEvent.change(screen.getByPlaceholderText(i18n.t.cron.namePlaceholder), { target: { value: 'My digest' } })
  fireEvent.change(screen.getByPlaceholderText(i18n.t.cron.promptPlaceholder), {
    target: { value: 'Keep my English instruction.' }
  })
  fireEvent.change(screen.getByPlaceholderText(i18n.t.cron.namePlaceholder), { target: { value: 'bad\0name' } })
  fireEvent.click(screen.getByRole('button', { name: i18n.t.cron.createAction }))
  expect(await screen.findByText(zh('cron.nameNul'))).toBeTruthy()
  expect(request).not.toHaveBeenCalled()
  fireEvent.change(screen.getByPlaceholderText(i18n.t.cron.namePlaceholder), { target: { value: 'My digest' } })
  fireEvent.change(screen.getByPlaceholderText('∞'), { target: { value: '3' } })
  fireEvent.click(screen.getAllByRole('combobox')[0])
  fireEvent.click(screen.getByRole('option', { name: zh('cron.freqOnce') }))
  expect(screen.getByText(zh('cron.minutesFromNow'))).toBeTruthy()
  await act(() => i18n.setLocale('ja'))
  const ja = translateBotsIn('ja')
  expect(screen.getByText(ja('cron.minutesFromNow'))).toBeTruthy()
  fireEvent.click(screen.getAllByRole('combobox')[0])
  fireEvent.click(screen.getByRole('option', { name: ja('cron.freqDaily') }))
  expect(screen.getByDisplayValue('3')).toBeTruthy()
  expect(screen.getByText(ja('cron.stopAfter'))).toBeTruthy()
  expect(screen.getAllByRole('combobox')[1].textContent).toBe(
    new Intl.DateTimeFormat('ja', { hour: 'numeric', minute: '2-digit' }).format(new Date(2000, 0, 1, 9, 0))
  )
  fireEvent.click(screen.getByRole('button', { name: i18n.t.cron.createAction }))
  await waitFor(() =>
    expect(request).toHaveBeenCalledWith(
      'cron.manage',
      expect.objectContaining({
        action: 'add',
        name: '[bot:writer] My digest',
        schedule: '0 9 * * *',
        repeat: 3,
        profile: 'writer'
      })
    )
  )
  expect(request.mock.calls.find(([method]) => method === 'cron.manage')?.[1]).toEqual(
    expect.objectContaining({ prompt: expect.stringContaining('Keep my English instruction.') })
  )
})
it('localizes inspector chrome and known states while preserving job data and unknown status', async () => {
  mount(
    <RoutineDetailDialog
      job={{
        job_id: 'job-1',
        name: '[bot:writer] User title',
        prompt_preview: 'User instruction',
        schedule: '0 9 * * 1-5',
        deliver: 'bot-chat',
        model: 'provider/model',
        workdir: '/my/work',
        last_status: 'delivery_failed'
      }}
      onClose={() => undefined}
      open
    />
  )
  const zh = translateBotsIn('zh')
  expect(screen.getByText(zh('cron.detailDescription'))).toBeTruthy()
  expect(screen.getByText(zh('cron.deliveryFailed'))).toBeTruthy()
  await act(() => i18n.setLocale('zh-hant'))
  expect(screen.getByText(translateBotsIn('zh-hant')('cron.deliveryFailed'))).toBeTruthy()

  for (const value of ['User title', 'User instruction', '0 9 * * 1-5', 'bot-chat', 'provider/model', '/my/work']) {
    expect(screen.getByText(value)).toBeTruthy()
  }

  expect(request).not.toHaveBeenCalled()
})
