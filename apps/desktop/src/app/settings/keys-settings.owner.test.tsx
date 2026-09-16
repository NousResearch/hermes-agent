import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import type { HermesApiRequest } from '@/global'
import { $settingsScopeOverride as selection } from '@/store/settings-scope'

afterEach(() => {
  cleanup()
  setApiRequestConnection(null)
  vi.unstubAllGlobals()
})

it('pins a copy of the mounted key owner for a later credential save', async () => {
  const owner = { connectionId: 'local', profile: 'shared-name' }
  selection.set(owner)
  setApiRequestConnection('fixture-ambient')
  const calls: HermesApiRequest[] = []
  vi.stubGlobal('hermesDesktop', {
    api: vi.fn(async (request: HermesApiRequest) => {
      calls.push(request)

      if (request.path === '/api/profiles') {
        return { profiles: [] }
      }

      return request.method === 'PUT'
        ? { ok: true }
        : {
            FIXTURE_TOOL_KEY: {
              category: 'tool',
              description: 'Fixture tool',
              is_password: true,
              is_set: false,
              redacted_value: null,
              tools: [],
              url: '',
              advanced: false
            }
          }
    })
  })
  const { KeysSettings } = await import('./keys-settings')

  const { container } = render(
    <MemoryRouter>
      <KeysSettings view="tools" />
    </MemoryRouter>
  )

  await screen.findByText('FIXTURE TOOL')
  const input = container.querySelector('input[type="password"]')!
  fireEvent.focus(input)
  fireEvent.change(input, { target: { value: 'fixture-value' } })
  // A caller-owned object is not the immutable owner of an existing draft.
  owner.connectionId = 'fixture-remote'
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(calls.filter(request => request.method === 'PUT')).toHaveLength(1))
  expect(calls.filter(request => request.method === 'PUT')[0]).toMatchObject({
    connectionId: 'local',
    profile: 'shared-name',
    body: { key: 'FIXTURE_TOOL_KEY', value: 'fixture-value' }
  })
})
