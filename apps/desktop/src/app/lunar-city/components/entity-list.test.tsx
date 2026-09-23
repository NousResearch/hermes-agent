// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { EntityKey, LunarCitySnapshot, LunarEntity } from '../model'

import { EntityList, orderedEntities } from './entity-list'

function key(value: string): EntityKey {
  return value as EntityKey
}

function entity(
  id: string,
  destination: LunarEntity['destination'],
  overrides: Partial<LunarEntity> = {}
): LunarEntity {
  return {
    authority: 'authoritative',
    animation: 'work',
    destination,
    identity: {
      connectionId: 'local',
      kind: 'session',
      profile: 'default',
      sessionId: id
    },
    key: key(`session:${destination}:${id}`),
    observedAt: 1_700_000_000_000,
    ...overrides
  }
}

function snapshot(entities: readonly LunarEntity[]): LunarCitySnapshot {
  return {
    entities: new Map(entities.map(value => [value.key, value])),
    observedAt: 1_700_000_000_000,
    revision: 1,
    sources: []
  }
}

describe('EntityList', () => {
  it('orders entities by manifest district and stable typed entity key', () => {
    const projectB = entity('zeta', 'project')
    const garden = entity('alpha', 'garden')
    const projectA = entity('alpha', 'project')

    expect(orderedEntities(snapshot([projectB, garden, projectA])).map(value => value.key)).toEqual([
      projectA.key,
      projectB.key,
      garden.key
    ])
  })

  it('uses current camera order inside each manifest district without changing exact entity keys', () => {
    const behind = entity('behind', 'project')
    const ahead = entity('ahead', 'project')
    const garden = entity('garden', 'garden')

    expect(
      orderedEntities(snapshot([behind, garden, ahead]), undefined, [behind.key, ahead.key]).map(value => value.key)
    ).toEqual([behind.key, ahead.key, garden.key])
  })

  it('uses native buttons with exact entity identity and readable state, destination, and authority', () => {
    const pip = entity('pip', 'lab', { animation: 'work' })
    const onSelect = vi.fn()

    render(<EntityList onSelect={onSelect} selectedEntityKey={pip.key} snapshot={snapshot([pip])} />)

    const button = screen.getByRole('button', { name: /Pip.*Working.*Research Lab.*Authoritative/i })
    expect(button.getAttribute('aria-pressed')).toBe('true')

    fireEvent.click(button)
    expect(onSelect).toHaveBeenCalledWith(pip)
  })

  it('does not rely on color and exposes stale or unknown state as text', () => {
    const unavailable = entity('worker-2', 'unknown', { animation: 'unavailable', authority: 'stale' })

    render(<EntityList onSelect={vi.fn()} snapshot={snapshot([unavailable])} />)

    expect(screen.getByRole('button', { name: /Worker 2.*Unavailable.*Unknown destination.*Stale/i })).toBeTruthy()
    expect(screen.getByText(/stale/i)).toBeTruthy()
  })

  it('exposes exact typed identity fields so duplicate display IDs remain distinguishable', () => {
    const left = entity('same-session', 'project', {
      identity: {
        connectionId: 'Remote-A',
        kind: 'session',
        profile: 'Profile-A',
        sessionId: 'same-session'
      },
      key: key('session:remote-a:same-session')
    })

    const right = entity('same-session', 'project', {
      identity: {
        connectionId: 'Remote-B',
        kind: 'session',
        profile: 'Profile-B',
        sessionId: 'same-session'
      },
      key: key('session:remote-b:same-session')
    })

    const onSelect = vi.fn()
    render(<EntityList onSelect={onSelect} snapshot={snapshot([left, right])} />)

    const buttons = screen.getAllByRole('button')
    expect(buttons).toHaveLength(2)
    expect(buttons[0]?.getAttribute('aria-label')).toContain('connectionId Remote-A')
    expect(buttons[0]?.getAttribute('aria-label')).toContain('profile Profile-A')
    expect(buttons[0]?.getAttribute('aria-label')).toContain('sessionId same-session')
    expect(buttons[1]?.getAttribute('aria-label')).toContain('connectionId Remote-B')
    expect(buttons[1]?.getAttribute('aria-label')).toContain('profile Profile-B')
    expect(screen.getByText(/Identity:.*connectionId Remote-A.*sessionId same-session/i)).toBeTruthy()
    expect(screen.getByText(/Identity:.*connectionId Remote-B.*sessionId same-session/i)).toBeTruthy()

    fireEvent.click(buttons[1]!)
    expect(onSelect).toHaveBeenCalledWith(right)
    expect(onSelect).not.toHaveBeenCalledWith(left)
  })

  it('exposes configured title, handle, every group, source position, and unavailable status', () => {
    const worker = entity('worker', 'unavailable', {
      animation: 'unavailable',
      authority: 'stale',
      identity: { connectionId: 'desktop-source', kind: 'profile', profile: 'test-contract-steward' },
      presentation: {
        configuredTitle: 'Test Contract Steward',
        groups: [
          { id: 'engineering', name: 'Engineering Guild' },
          { id: 'release', name: 'Acceptance & Release' }
        ],
        metadata: { source: 'profiles:desktop-source', state: 'unavailable' },
        placement: { lodHint: 0, overflow: false, primaryGroupId: 'engineering', slot: 1 },
        profileHandle: '@test-contract-steward',
        sourceLabel: 'Hermes Desktop'
      }
    })

    render(<EntityList onSelect={vi.fn()} snapshot={snapshot([worker])} />)

    const button = screen.getByRole('button', { name: /Test Contract Steward.*@test-contract-steward/i })
    expect(button.getAttribute('aria-label')).toContain('Groups: Engineering Guild, Acceptance & Release')
    expect(button.getAttribute('aria-label')).toContain('Source: Hermes Desktop')
    expect(button.getAttribute('aria-label')).toContain('Profile metadata: Unavailable')
    expect(button.getAttribute('aria-label')).toContain('connectionId desktop-source')
    expect(button.getAttribute('aria-label')).toContain('Unavailable')
  })
})
