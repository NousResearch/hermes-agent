import { host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import type { ButtonHTMLAttributes } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $canonicalGroupBindings, CanonicalGroupList } from './canonical-group-registry'

const { request } = vi.hoisted(() => ({ request: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')
  const { useStore } = await import('@nanostores/react')

  return {
    atom, useValue: useStore,
    Button: ({ variant: _variant, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: string }) => <button {...props} />,
    host: {
      requestProfile: request,
      state: { connectionId: atom<string | null>(null), profile: atom('default'), gateway: atom('closed') }
    }
  }
})
vi.mock('./canonical-group-labels', () => ({
  useCanonicalGroupLabels: () => ({ refreshGroups: 'Refresh gateway groups' })
}))

const state = {
  connectionId: host.state.connectionId as WritableAtom<string | null>,
  gateway: host.state.gateway as WritableAtom<string>
}

beforeEach(() => {
  state.connectionId.set(null)
  state.gateway.set('closed')
  $canonicalGroupBindings.set({})
  request.mockReset().mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {return { driver: true }}

    return { rooms: [{ room_id: 'team', name: 'Team', members: [] }], next_offset: null }
  })
})
afterEach(cleanup)

it('waits for connection readiness, then loads groups without a manual refresh', async () => {
  await act(async () => { render(<CanonicalGroupList onOpen={vi.fn()} />) })
  expect(screen.queryByRole('alert')).toBeNull()
  await act(async () => { state.connectionId.set('local') })
  expect(request).not.toHaveBeenCalled()
  await act(async () => { state.gateway.set('open') })
  expect(screen.getByRole('button', { name: 'Team' })).toBeTruthy()
  expect(request.mock.calls.map(call => call[1])).toEqual(['groups.capabilities', 'groups.list'])
})

it('discards a disconnected read, retries on reconnect, and still exposes real server errors', async () => {
  state.connectionId.set('local')
  state.gateway.set('open')
  let finish!: (value: unknown) => void
  request.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  await act(async () => { render(<CanonicalGroupList onOpen={vi.fn()} />) })
  await act(async () => { state.gateway.set('closed') })
  await act(async () => { finish({ driver: true }) })
  expect(screen.queryByRole('button', { name: 'Team' })).toBeNull()
  expect($canonicalGroupBindings.get()).toEqual({})

  request.mockRejectedValueOnce(new Error('Group access denied'))
  await act(async () => { state.gateway.set('open') })
  expect(screen.getByRole('alert').textContent).toBe('Group access denied')
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Refresh gateway groups' })) })
  expect(screen.queryByRole('alert')).toBeNull()
  expect(screen.getByRole('button', { name: 'Team' })).toBeTruthy()
})
