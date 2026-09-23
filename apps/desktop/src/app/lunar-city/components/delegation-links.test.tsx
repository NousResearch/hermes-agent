// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { entityKey } from '../identity'
import type { LunarCitySnapshot, LunarEntity } from '../model'

import { DelegationLinks } from './delegation-links'

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})
it('draws at most eight exact links, exposes equivalent inspection, and stops sampling when hidden or reduced', () => {
  vi.useFakeTimers()
  const identity = { kind: 'session', connectionId: 'local', profile: 'p', sessionId: 's' } as const

  const parent: LunarEntity = {
    identity,
    key: entityKey(identity),
    authority: 'authoritative',
    destination: 'project',
    sourceState: 'running',
    animation: 'work',
    observedAt: 1
  }

  const children = Array.from({ length: 10 }, (_, i) => {
    const childIdentity = { ...identity, kind: 'subagent', subagentId: String(i) } as const

    return { ...parent, identity: childIdentity, key: entityKey(childIdentity) }
  })

  const snapshot: LunarCitySnapshot = {
    revision: 1,
    observedAt: 1,
    sources: [],
    entities: new Map([parent, ...children].map(entity => [entity.key, entity]))
  }

  const presentation = vi.fn(() => ({ position: { x: 1, y: 2, z: 3 }, moving: true, animation: 'walk' }))

  const props = {
    snapshot,
    visible: true,
    enabled: true,
    reducedMotion: false,
    presentation,
    project: () => ({ x: 10, y: 20 }),
    onSelectEntity: vi.fn()
  }

  const { container, rerender } = render(<DelegationLinks {...props} />)
  expect(container.querySelectorAll('path')).toHaveLength(0)
  act(() => vi.advanceTimersByTime(100))
  expect(container.querySelectorAll('path')).toHaveLength(8)
  fireEvent.click(screen.getByRole('button', { name: 'Inspect subagent 0 · p · local' }))
  expect(props.onSelectEntity).toHaveBeenCalledWith(children[0])
  act(() => vi.advanceTimersByTime(100))
  expect(vi.getTimerCount()).toBe(1)
  rerender(<DelegationLinks {...props} reducedMotion />)
  expect(container.querySelectorAll('path')).toHaveLength(0)
  expect(vi.getTimerCount()).toBe(0)
  rerender(<DelegationLinks {...props} visible={false} />)
  const calls = presentation.mock.calls.length
  act(() => vi.advanceTimersByTime(1000))
  expect(presentation).toHaveBeenCalledTimes(calls)
})

it('waits for delayed navigation, retries briefly after a stop, and leaves no timer for ineligible or stationary edges', () => {
  vi.useFakeTimers()
  const identity = { kind: 'session', connectionId: 'local', profile: 'p', sessionId: 's' } as const

  const parent: LunarEntity = {
    identity,
    key: entityKey(identity),
    authority: 'authoritative',
    destination: 'project',
    sourceState: 'running',
    animation: 'work',
    observedAt: 1
  }

  const childIdentity = { ...identity, kind: 'subagent', subagentId: 'c' } as const
  const child = { ...parent, identity: childIdentity, key: entityKey(childIdentity) }

  const snapshot: LunarCitySnapshot = {
    revision: 1,
    observedAt: 1,
    sources: [],
    entities: new Map([parent, child].map(entity => [entity.key, entity]))
  }

  let moving = false
  const presentation = vi.fn(() => ({ position: { x: 1, y: 2, z: 3 }, moving, animation: 'walk' }))

  const props = {
    snapshot,
    visible: true,
    enabled: true,
    reducedMotion: false,
    presentation,
    project: () => ({ x: 10, y: 20 }),
    onSelectEntity: vi.fn()
  }

  const { container, rerender } = render(<DelegationLinks {...props} />)
  act(() => vi.advanceTimersByTime(200))
  expect(container.querySelectorAll('path')).toHaveLength(0)
  moving = true
  act(() => vi.advanceTimersByTime(100))
  expect(container.querySelectorAll('path')).toHaveLength(1)
  moving = false
  act(() => vi.advanceTimersByTime(100))
  moving = true
  act(() => vi.advanceTimersByTime(100))
  expect(container.querySelectorAll('path')).toHaveLength(1)
  moving = false
  act(() => vi.advanceTimersByTime(3100))
  expect(vi.getTimerCount()).toBe(0)
  rerender(<DelegationLinks {...props} snapshot={{ ...snapshot, entities: new Map([[parent.key, parent]]) }} />)
  expect(vi.getTimerCount()).toBe(0)
})
