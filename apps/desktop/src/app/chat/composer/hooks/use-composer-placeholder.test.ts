import { renderHook } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { beginBackendTransition, resetBackendTransitionsForTests } from '@/store/backend-transition'
import { $activeGatewayProfile } from '@/store/profile'

import { useComposerPlaceholder } from './use-composer-placeholder'

afterEach(() => {
  resetBackendTransitionsForTests()
  $activeGatewayProfile.set('default')
})

it("names the profile's sandbox restart instead of 'Reconnecting to Hermes…' while the socket is down", () => {
  $activeGatewayProfile.set('work')
  beginBackendTransition('work', 'restarting', { source: 'launcher' })

  const { result, rerender } = renderHook(props => useComposerPlaceholder(props), {
    initialProps: { disabled: true, reconnecting: true, sessionId: 's1' } as Parameters<
      typeof useComposerPlaceholder
    >[0]
  })

  expect(result.current).toBe('Restarting sandbox…')

  rerender({ disabled: true, profile: 'default', reconnecting: true, sessionId: 's1' })
  expect(result.current).toBe('Reconnecting to Hermes…')
})

it("names the profile's sandbox start while the connection is still open", () => {
  beginBackendTransition('work', 'starting', { source: 'switch' })

  const { result, rerender } = renderHook(props => useComposerPlaceholder(props), {
    initialProps: { disabled: false, profile: 'work', reconnecting: false, sessionId: 's1' } as Parameters<
      typeof useComposerPlaceholder
    >[0]
  })

  expect(result.current).toBe('Starting sandbox…')

  rerender({ disabled: false, profile: 'default', reconnecting: false, sessionId: 's1' })
  expect(result.current).not.toBe('Starting sandbox…')
})
