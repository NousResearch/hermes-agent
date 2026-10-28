import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { useEnvCredentials } from './env-credentials'

vi.mock('@/hermes', () => ({
  getEnvVars: async () => ({}),
  revealEnvVar: async () => ({ value: 'test-secret' }),
  deleteEnvVar: vi.fn(),
  setEnvVar: vi.fn()
}))

afterEach(cleanup)

it.each([
  { profile: 'beta', section: 'keys' },
  { profile: 'alpha', section: 'accounts' }
])('hides secrets when the credential scope changes to $profile/$section', async next => {
  const { result, rerender } = renderHook(
    ({ profile, section }) => useEnvCredentials(profile, section),
    { initialProps: { profile: 'alpha', section: 'keys' } }
  )
  await act(async () => { await result.current.rowProps.onReveal('KEY') })
  expect(result.current.rowProps.revealed).toEqual({ KEY: 'test-secret' })
  await act(async () => { rerender(next) })
  expect(result.current.rowProps.revealed).toEqual({})
})
