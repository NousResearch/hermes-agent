import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import {
  activateCustomEndpoint,
  deleteCustomEndpoint,
  getCustomEndpoints,
  saveCustomEndpoint,
  validateCustomEndpoint
} from '@/hermes'
import { I18nProvider } from '@/i18n'
import { ko } from '@/i18n/ko'
import { settingsRiskCopyKo } from '@/i18n/settings-risk-copy'
import { $confirmRequest, settleConfirm } from '@/store/confirm'
import { $notifications, clearNotifications } from '@/store/notifications'
import type { CustomEndpoint } from '@/types/hermes'

import { CustomEndpointsSettings } from './custom-endpoints-settings'

vi.mock('@/hermes', () => ({
  activateCustomEndpoint: vi.fn(),
  deleteCustomEndpoint: vi.fn(),
  getCustomEndpoints: vi.fn(),
  saveCustomEndpoint: vi.fn(),
  validateCustomEndpoint: vi.fn()
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))

const copy = settingsRiskCopyKo.customEndpoints

const endpoint: CustomEndpoint = {
  id: 'audit-endpoint',
  name: '검증 엔드포인트',
  base_url: 'https://audit.invalid/v1',
  model: 'audit-model',
  models: ['audit-model'],
  has_api_key: true,
  discover_models: true,
  is_current: false
}

const response = { endpoints: [endpoint], current: { provider: '', model: '', base_url: '' } }

function mount() {
  render(
    <I18nProvider configClient={null} initialLocale="ko">
      <CustomEndpointsSettings />
    </I18nProvider>
  )
}

beforeEach(() => {
  vi.resetAllMocks()
  vi.mocked(getCustomEndpoints).mockResolvedValue(response)
  vi.mocked(saveCustomEndpoint).mockResolvedValue({ ...response, id: endpoint.id })
})

afterEach(() => {
  cleanup()
  settleConfirm(false)
  clearNotifications()
})

it('preserves the saved key with a blank edit and separates model-list validation from saving or activating', async () => {
  mount()
  const key = await screen.findByLabelText(copy.apiKey)
  expect(key.getAttribute('placeholder')).toBe(copy.keepCurrentKey)
  expect(screen.getByText(copy.testHint)).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: ko.common.save }))
  await waitFor(() => expect(saveCustomEndpoint).toHaveBeenCalledOnce())
  const payload = vi.mocked(saveCustomEndpoint).mock.calls[0][0]
  expect(payload.api_key).toBeUndefined()
  expect(JSON.parse(JSON.stringify(payload))).not.toHaveProperty('api_key')
  expect(payload).toMatchObject({ id: endpoint.id, make_default: false, base_url: endpoint.base_url })
  expect(activateCustomEndpoint).not.toHaveBeenCalled()

  vi.mocked(validateCustomEndpoint).mockResolvedValue({
    ok: false,
    reachable: true,
    models: [],
    message: 'HTTP 401: fixture key rejected'
  })
  fireEvent.click(screen.getByRole('button', { name: copy.test }))
  await waitFor(() => expect(validateCustomEndpoint).toHaveBeenCalledOnce())
  expect(vi.mocked(validateCustomEndpoint).mock.calls[0][0].api_key).toBeUndefined()
  await waitFor(() =>
    expect($notifications.get().at(-1)).toMatchObject({
      kind: 'warning',
      message: copy.validationFailed,
      detail: 'HTTP 401: fixture key rejected'
    })
  )
  expect(saveCustomEndpoint).toHaveBeenCalledOnce()
})

it('confirms endpoint deletion in the current language without implying remote service or key revocation', async () => {
  mount()
  const remove = await screen.findByRole('button', { name: ko.settings.customEndpoints.deleteEndpoint })
  fireEvent.click(remove)
  expect($confirmRequest.get()).toMatchObject({
    title: copy.deleteConfirm(endpoint.name),
    description: copy.deleteDescription,
    confirmLabel: ko.common.delete,
    cancelLabel: ko.common.cancel,
    destructive: true
  })
  await act(async () => settleConfirm(false))
  expect(deleteCustomEndpoint).not.toHaveBeenCalled()
  vi.mocked(deleteCustomEndpoint).mockResolvedValue({ ...response, endpoints: [] })
  fireEvent.click(remove)
  await act(async () => settleConfirm(true))
  expect(deleteCustomEndpoint).toHaveBeenCalledExactlyOnceWith(endpoint.id)
  expect(await screen.findByText(ko.settings.customEndpoints.emptyTitle)).toBeTruthy()
  expect((screen.getByRole('textbox', { name: copy.name }) as HTMLInputElement).value).toBe('')
})
