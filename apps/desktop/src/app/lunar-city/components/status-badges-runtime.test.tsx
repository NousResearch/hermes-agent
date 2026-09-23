// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { Profiler } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import type { EntityKey, LunarCityWorldHandle, LunarEntity } from '../model'
import { $lunarCitySnapshot } from '../store'

import { LunarCityStatusBadges } from './status-badges'

afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks() })

it('keeps unchanged pins render-idle, selects the current source record, and suspends hidden sampling', () => {
  vi.useFakeTimers()
  const hidden = vi.spyOn(window.document, 'hidden', 'get').mockReturnValue(false)

  const entity: LunarEntity = {
    authority: 'authoritative', animation: 'idle', destination: 'lab',
    identity: { connectionId: 'local', kind: 'profile', profile: 'worker' },
    key: 'worker' as EntityKey, observedAt: 1, position: { x: 1, y: 0, z: 1 }, sourceState: 'working'
  }

  const publish = (worker: LunarEntity) => $lunarCitySnapshot.set({ entities: new Map([[worker.key, worker]]), observedAt: worker.observedAt, revision: worker.observedAt, sources: [] })
  publish(entity)
  const project = vi.fn(() => ({ x: 25, y: 30, visible: true }))
  const onSelect = vi.fn(), commits = vi.fn()
  const view = render(<Profiler id="pins" onRender={commits}><LunarCityStatusBadges onSelect={onSelect} qualityTier="balanced" rendererStatus="ready" worldRef={{ current: { projectWorldPoint: project } as unknown as LunarCityWorldHandle }} /></Profiler>)
  commits.mockClear()
  const current = { ...entity, observedAt: 2 }
  act(() => { publish(current); vi.advanceTimersByTime(1000) })
  expect(commits).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button'))
  expect(onSelect).toHaveBeenCalledWith(current)
  hidden.mockReturnValue(true)
  act(() => window.document.dispatchEvent(new Event('visibilitychange')))
  project.mockClear()
  act(() => vi.advanceTimersByTime(10000))
  expect(project).not.toHaveBeenCalled()
  expect(vi.getTimerCount()).toBe(0)
  hidden.mockReturnValue(false)
  act(() => window.document.dispatchEvent(new Event('visibilitychange')))
  expect(screen.getByRole('button')).toBeTruthy()
  expect(project).toHaveBeenCalledOnce()
  view.unmount()
  expect(vi.getTimerCount()).toBe(0)
})
