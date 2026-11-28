import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, renderHook, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ReasoningPill } from '@/app/chat/composer/reasoning-pill'
import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { useModelMenuController } from '@/app/shell/use-model-menu-controller'
import { createClientSessionState } from '@/lib/chat-runtime'
import { getComposerReasoningEffort, setComposerReasoningEffort } from '@/store/composer-reasoning-effort'
import { resetBinding, setBinding } from '@/store/keybinds'
import {
  $activeSessionId,
  $currentReasoningEffort,
  $selectedStoredSessionId,
  setComposerSelectionOwner,
  setCurrentReasoningEffort,
  setDefaultReasoningEffort
} from '@/store/session'
import { $sessionStates, clearAllSessionStates, sessionTileDelegate } from '@/store/session-states'
import { deferred } from '@/test/deferred'

import { type KeybindRuntimeDeps, useKeybinds } from './use-keybinds'

vi.mock('@/store/session-states', async original => ({
  ...(await original<Record<string, unknown>>()),
  sessionTileDelegate: vi.fn(() => null)
}))

vi.mock('@/themes/context', () => ({ useTheme: () => ({ resolvedMode: 'light', setMode: vi.fn() }) }))

function setup(
  requestGateway: KeybindRuntimeDeps['requestGateway'] = vi.fn(async () => ({ value: 'medium' }) as never)
) {
  renderHook(
    () =>
      useKeybinds({
        archiveSelectedSession: vi.fn(),
        openNewSessionTab: vi.fn(),
        requestGateway,
        startFreshSession: vi.fn(),
        toggleCommandCenter: vi.fn(),
        toggleSelectedPin: vi.fn()
      }),
    { wrapper: ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter> }
  )
  render(
    <ReasoningPill
      disabled={false}
      model={{
        canSwitch: true,
        model: 'example-model',
        provider: 'example',
        reasoningMenuContent: <div>menu</div>
      }}
    />
  )

  return requestGateway
}

function press(direction: 'up' | 'down') {
  act(() => {
    const event = new KeyboardEvent('keydown', {
      altKey: true,
      bubbles: true,
      cancelable: true,
      code: direction === 'up' ? 'ArrowUp' : 'ArrowDown',
      key: direction === 'up' ? 'ArrowUp' : 'ArrowDown'
    })

    window.dispatchEvent(event)
    expect(event.defaultPrevented).toBe(true)
  })
}

beforeEach(() => {
  vi.mocked(sessionTileDelegate).mockReturnValue(null)
  clearAllSessionStates()
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  setComposerSelectionOwner('example', 'default')
  setComposerReasoningEffort('')
  setCurrentReasoningEffort('low')
  setDefaultReasoningEffort('high')
  setBinding('composer.reasoningDown', ['alt+down'])
  setBinding('composer.reasoningUp', ['alt+up'])
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  resetBinding('composer.reasoningDown')
  resetBinding('composer.reasoningUp')
})

describe('reasoning shortcuts', () => {
  it.each([
    ['', 'down', 'Med', 'medium'],
    ['none', 'up', 'Min', 'minimal']
  ] as const)('steps the visible draft pick %s and supplies the create getter', (pick, direction, label, effort) => {
    setComposerReasoningEffort(pick)
    const request = setup()
    press(direction)
    expect.soft(screen.getByTestId('reasoning-pill').textContent).toBe(label)
    expect(getComposerReasoningEffort()).toBe(effort)
    expect(request).not.toHaveBeenCalled()
  })

  it.each(['', 'none'])('rolls back a failed live edit to its mirror and deliberate pick (%s)', async pick => {
    setComposerReasoningEffort(pick)
    $activeSessionId.set('runtime-a')
    setCurrentReasoningEffort('high')
    const state = { ...createClientSessionState(null), reasoningEffort: 'high' }
    $sessionStates.set({ 'runtime-a': state })
    vi.mocked(sessionTileDelegate).mockReturnValue({
      updateSession: (id, update) => {
        const next = update($sessionStates.get()[id])
        $sessionStates.set({ ...$sessionStates.get(), [id]: next })

        return next
      }
    } as ReturnType<typeof sessionTileDelegate>)
    const write = deferred<never>()
    const request = setup(vi.fn(() => write.promise))
    press('down')
    expect.soft(getComposerReasoningEffort()).toBe('medium')
    expect(screen.getByTestId('reasoning-pill').textContent).toBe('Med')
    await act(async () => {})
    expect(request).toHaveBeenCalledWith('config.set', { key: 'reasoning', session_id: 'runtime-a', value: 'medium' })
    await act(async () => {
      write.reject(new Error('fixture failure'))
      await write.promise.catch(() => {})
    })
    expect(getComposerReasoningEffort()).toBe(pick)
    expect($currentReasoningEffort.get()).toBe('high')
    expect(PRIMARY_SESSION_VIEW.$reasoningEffort.get()).toBe('high')
  })

  it('keeps a newer draft pick when an old live request settles', async () => {
    $activeSessionId.set('runtime-a')
    setCurrentReasoningEffort('high')
    const write = deferred<never>()
    setup(vi.fn(() => write.promise))
    press('down')
    act(() => {
      $activeSessionId.set(null)
      setComposerSelectionOwner('example', 'other')
      setComposerReasoningEffort('none')
      setCurrentReasoningEffort('none')
    })
    press('up')
    await act(async () => {
      write.reject(new Error('fixture failure'))
      await write.promise.catch(() => {})
    })
    expect(getComposerReasoningEffort()).toBe('minimal')
    expect(screen.getByTestId('reasoning-pill').textContent).toBe('Min')
  })
})

describe('overlapping live reasoning shortcuts', () => {
  it.each(['effort', 'effort-then-keyboard', 'speed'] as const)(
    'only a newer reasoning menu intent supersedes queued shortcuts (%s)',
    async menuIntent => {
      const laterKeyboard = menuIntent === 'effort-then-keyboard'
      const speedOnly = menuIntent === 'speed'
      $activeSessionId.set('runtime-a')
      setCurrentReasoningEffort('high')
      const first = deferred<never>()
      const values: string[] = []
      let backendEffort = 'high'

      const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
        if (method !== 'config.set') {
          return { providers: [] } as never
        }

        const value = String(params?.value)

        if (params?.key !== 'reasoning') {
          return { value } as never
        }

        values.push(value)
        backendEffort = value

        if (values.length === 1) {
          await first.promise
        }

        return { value } as never
      })

      setup(request)
      const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

      const { result } = renderHook(() => useModelMenuController({ requestGateway: request, onSelectModel: vi.fn() }), {
        wrapper: ({ children }: { children: ReactNode }) => (
          <QueryClientProvider client={client}>{children}</QueryClientProvider>
        )
      })

      press('down')
      await waitFor(() => expect(values).toEqual(['medium']))
      press('down')
      act(() =>
        result.current.controller.setOptions(speedOnly ? { fast: true } : { effort: 'none' }, {
          model: 'example-model',
          provider: 'example',
          isActive: true
        })
      )
      await waitFor(() => expect(values).toEqual(speedOnly ? ['medium'] : ['medium', 'none']))

      if (laterKeyboard) {
        press('up')
        await waitFor(() => expect(values).toEqual(['medium', 'none', 'minimal']))
      }

      await act(async () => {
        first.resolve({ value: 'medium' } as never)
        await first.promise
      })
      expect(values).toEqual(
        speedOnly ? ['medium', 'low'] : laterKeyboard ? ['medium', 'none', 'minimal'] : ['medium', 'none']
      )
      expect(backendEffort).toBe(speedOnly ? 'low' : laterKeyboard ? 'minimal' : 'none')
      expect(getComposerReasoningEffort()).toBe(speedOnly ? 'low' : laterKeyboard ? 'minimal' : 'none')
      client.clear()
    }
  )

  it.each([null, 'low'])('restores confirmed effort after repeated same-value edits (first reply: %s)', async reply => {
    $activeSessionId.set('runtime-a')
    setCurrentReasoningEffort('high')
    const writes = [deferred<never>(), deferred<never>(), deferred<never>()]

    const request = setup(
      vi
        .fn()
        .mockImplementationOnce(() => writes[0].promise)
        .mockImplementationOnce(() => writes[1].promise)
        .mockImplementationOnce(() => writes[2].promise)
    )

    press('down')
    press('up')
    press('down')
    await waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    await act(async () => {
      if (reply) {
        writes[0].resolve({ value: reply } as never)
      } else {
        writes[0].reject(new Error('fixture failure'))
      }

      await writes[0].promise.catch(() => {})
    })
    expect(screen.getByTestId('reasoning-pill').textContent).toBe('Med')

    for (const index of [1, 2]) {
      await waitFor(() => expect(request).toHaveBeenCalledTimes(index + 1))
      await act(async () => {
        writes[index].reject(new Error('fixture failure'))
        await writes[index].promise.catch(() => {})
      })
    }

    expect(getComposerReasoningEffort()).toBe(reply ?? '')
    expect($currentReasoningEffort.get()).toBe(reply ?? 'high')
    expect(screen.getByTestId('reasoning-pill').textContent).toBe(reply ? 'Low' : 'High')
  })

  it.each([false, true])('does not repaint a different profile after an old request settles (%s)', async success => {
    $activeSessionId.set('runtime-a')
    setCurrentReasoningEffort('high')
    const write = deferred<never>()
    const request = setup(vi.fn(() => write.promise))
    press('down')
    await waitFor(() => expect(request).toHaveBeenCalledOnce())
    act(() => {
      setComposerSelectionOwner('example', 'other')
      setComposerReasoningEffort('none')
      setCurrentReasoningEffort('none')
    })
    await act(async () => {
      if (success) {
        write.resolve({ value: 'medium' } as never)
      } else {
        write.reject(new Error('fixture failure'))
      }

      await write.promise.catch(() => {})
    })
    expect(getComposerReasoningEffort()).toBe('none')
    expect($currentReasoningEffort.get()).toBe('none')
  })
})
