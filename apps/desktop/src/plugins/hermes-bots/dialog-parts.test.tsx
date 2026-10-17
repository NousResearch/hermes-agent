import type * as HermesSdk from '@hermes/plugin-sdk'
import type { PluginContext } from '@hermes/plugin-sdk'
import { Input, Textarea } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

// The harness supplies the host's provider and plugin registry.
// eslint-disable-next-line no-restricted-imports
import { createPluginI18n, I18nProvider } from '@/i18n'

import { CreateAgentDialog } from './create-dialog'
import { labeled } from './dialog-parts'
import { BOTS_LOCALES } from './i18n'
import { setPluginCtx } from './shared'

const mocks = vi.hoisted(() => ({
  connections: vi.fn(async () => []),
  request: vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({}))
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, host: { ...sdk.host, connections: mocks.connections, request: mocks.request } }
})

let disposeLocales: () => void

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

beforeEach(() => {
  vi.clearAllMocks()
  const i18n = createPluginI18n('hermes-bots', dispose => dispose)
  disposeLocales = i18n.register(BOTS_LOCALES)
  setPluginCtx({ i18n } as PluginContext)
})

afterEach(() => {
  cleanup()
  disposeLocales()
  setPluginCtx(null)
})

it('names the real New Bot text controls by their visible Korean labels', async () => {
  await act(async () => {
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <I18nProvider configClient={null} initialLocale="ko">
          <CreateAgentDialog onClose={() => undefined} open roster={[{ name: 'default' }]} />
        </I18nProvider>
      </QueryClientProvider>
    )
  })

  for (const caption of ['이름', '표시 제목', '설명']) {
    const input = screen.getByRole('textbox', { name: caption })
    const label = screen.getByText(caption, { selector: 'label', exact: true }) as HTMLLabelElement

    expect(label.control).toBe(input)
  }

  expect(mocks.request.mock.calls.some(([method]) => method === 'profiles.create')).toBe(false)
})

it('keeps each label on its own input across rerenders and preserves caller IDs and click handlers', () => {
  const onDescriptionClick = vi.fn()

  const form = (description: string) => (
    <>
      {labeled('이름', <Input id="existing-profile-name" />)}
      {labeled(description, <Textarea onClick={onDescriptionClick} />)}
      {labeled('다른 이름', <Input />)}
    </>
  )

  const view = render(form('설명'))
  const name = screen.getByRole('textbox', { name: '이름' })
  const description = screen.getByRole('textbox', { name: '설명' })
  const otherName = screen.getByRole('textbox', { name: '다른 이름' })
  const ids = [name.id, description.id, otherName.id]

  expect(name.id).toBe('existing-profile-name')
  expect(ids.every(Boolean)).toBe(true)
  expect(new Set(ids).size).toBe(ids.length)
  fireEvent.click(screen.getByText('설명', { selector: 'label' }))
  expect(onDescriptionClick).toHaveBeenCalledTimes(1)

  view.rerender(form('Description'))
  expect(screen.getByRole('textbox', { name: 'Description' })).toBe(description)
  expect([name.id, description.id, otherName.id]).toEqual(ids)
  expect((screen.getByText('Description', { selector: 'label' }) as HTMLLabelElement).control).toBe(description)
})
