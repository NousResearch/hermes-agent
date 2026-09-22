// @vitest-environment jsdom
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { EntityKey, LunarCityWorldHandle } from '../model'
import type { WorkerEncounter } from '../worker-social'

import { WorkerLife } from './worker-life'
afterEach(()=>{cleanup();vi.useRealTimers();vi.restoreAllMocks()})
it('samples copied phases, stops all hidden polling, and resumes without stale encounter text',()=>{
 vi.useFakeTimers()
 const encounter:WorkerEncounter={id:'pair',participants:['one','two'] as EntityKey[] as [EntityKey,EntityKey],phase:'greeting',elapsedMs:0,provenance:'local-ambient'}
 const read=vi.fn(()=>[encounter]),hidden=vi.spyOn(window.document,'hidden','get').mockReturnValue(false)
 render(<WorkerLife onSelect={vi.fn()} snapshot={{revision:1,observedAt:1,entities:new Map(),sources:[]}} worldRef={{current:{getWorkerEncounters:read} as unknown as LunarCityWorldHandle}}/>)
 expect(screen.getByText(': greeting')).toBeTruthy()
 encounter.phase='exchange'
 act(()=>vi.advanceTimersByTime(1000))
 expect(screen.getByText(': exchange')).toBeTruthy()
 hidden.mockReturnValue(true)
 act(()=>window.document.dispatchEvent(new Event('visibilitychange')))
 read.mockClear()
 act(()=>vi.advanceTimersByTime(10000))
 expect(read).not.toHaveBeenCalled();expect(vi.getTimerCount()).toBe(0)
 read.mockReturnValue([])
 hidden.mockReturnValue(false)
 act(()=>window.document.dispatchEvent(new Event('visibilitychange')))
 expect(screen.getByText('No local conversations active.')).toBeTruthy()
})
