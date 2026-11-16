import { useStore } from '@nanostores/react'
import { act } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { reactRoot } from '@/test/react-root'

vi.mock('@/store/session', async () => {
  const { atom } = await import('nanostores')

  return { $activeSessionId: atom('current'), $currentCwd: atom('') }
})
vi.mock('@/store/composer-status', async () => ({
  $backgroundStatusBySession: (await import('nanostores')).atom({})
}))
// Keep the real workspace and terminal stores; replace only the native renderer/PTY boundary.
vi.mock('./instance', () => ({
  AgentTerminalInstance: ({ active, procId }: { active: boolean; procId: string }) => (
    <div data-active={active} data-agent={procId} />
  ),
  TerminalInstance: () => <div data-user-terminal="" />
}))

const mount = reactRoot()

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
  vi.spyOn(window.document, 'hasFocus').mockReturnValue(true)
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
    top: 0,
    left: 0,
    right: 640,
    bottom: 320,
    width: 640,
    height: 320
  } as DOMRect)
})

afterEach(() => {
  mount.unmount()
  vi.restoreAllMocks()
})

async function setup(preference: string) {
  window.localStorage.setItem('hermes.desktop.revealBackgroundTerminals', preference)
  const { PersistentTerminal, TerminalSlot } = await import('./persistent')
  const { $terminalTakeover, setTerminalTakeover } = await import('../store')
  const { $backgroundStatusBySession } = await import('@/store/composer-status')
  const { $activeSessionId } = await import('@/store/session')
  const terminals = await import('./terminals')

  $backgroundStatusBySession.set({})
  $activeSessionId.set('current')

  function Harness() {
    const open = useStore($terminalTakeover)

    return (
      <>
        {open && <TerminalSlot />}
        <PersistentTerminal onAddSelectionToChat={() => undefined} />
      </>
    )
  }

  mount.render(<Harness />)

  const publish = (session: string, id: string, output = '') =>
    act(() => {
      $backgroundStatusBySession.set({
        ...$backgroundStatusBySession.get(),
        [session]: [{ id, title: `command ${id}`, output, type: 'background', state: 'running' }]
      })
    })

  return { ...terminals, $activeSessionId, $terminalTakeover, publish, setTerminalTakeover }
}

it('reveals before the first terminal mount and handles hidden-pane task lifecycle without user shells', async () => {
  vi.mocked(HTMLElement.prototype.getBoundingClientRect).mockReturnValue({
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    width: 0,
    height: 0
  } as DOMRect)
  const state = await setup('auto')
  expect(mount.container!.querySelector('[data-agent]')).toBeNull()
  expect(state.$terminals.get()).toEqual([])

  state.publish('current', 'first')
  expect(state.$terminalTakeover.get()).toBe(true)
  expect(state.$terminals.get().find(term => term.id === state.$activeTerminalId.get())?.procId).toBe('first')
  // Takeover creates the slot, but zero dimensions must not boot xterm/PTYs.
  expect(mount.container!.querySelector('[data-agent]')).toBeNull()
  vi.mocked(HTMLElement.prototype.getBoundingClientRect).mockReturnValue({
    top: 0,
    left: 0,
    right: 640,
    bottom: 320,
    width: 640,
    height: 320
  } as DOMRect)
  await act(async () => {
    window.dispatchEvent(new Event('resize'))
    await new Promise<void>(resolve => window.requestAnimationFrame(() => resolve()))
  })
  expect(mount.container!.querySelector('[data-agent="first"]')).not.toBeNull()
  expect(state.$terminals.get().some(term => term.kind === 'user')).toBe(false)
  expect(mount.container!.querySelector('[data-user-terminal]')).toBeNull()

  act(() => state.setTerminalTakeover(false))
  state.publish('current', 'first', 'updated')
  expect(state.$terminalTakeover.get()).toBe(false)
  state.publish('other', 'other-task')
  expect(state.$terminalTakeover.get()).toBe(false)
  act(() => state.$activeSessionId.set('other'))
  expect(state.$terminalTakeover.get()).toBe(false)
  act(() => state.$activeSessionId.set('current'))
  expect(state.$terminalTakeover.get()).toBe(false)

  act(() => state.closeAgentTerminalByProc('first'))
  state.publish('current', 'first', 'after close')
  expect(state.$terminals.get().some(term => term.procId === 'first')).toBe(false)
  expect(state.$terminalTakeover.get()).toBe(false)

  state.publish('current', 'next')
  expect(state.$terminalTakeover.get()).toBe(true)
  expect(state.$terminals.get().find(term => term.id === state.$activeTerminalId.get())?.procId).toBe('next')
  expect(mount.container!.querySelector('[data-agent="next"]')).not.toBeNull()
  expect(state.$terminals.get().some(term => term.kind === 'user')).toBe(false)
})

it('surfaces tasks without takeover or terminal mounting for stack and invalid preferences', async () => {
  for (const preference of ['stack', 'invalid']) {
    const state = await setup(preference)
    state.publish('current', `task-${preference}`)
    expect(state.$terminals.get().some(term => term.procId === `task-${preference}`)).toBe(true)
    expect(state.$activeTerminalId.get()).toBeNull()
    expect(state.$terminalTakeover.get()).toBe(false)
    expect(mount.container!.querySelector('[data-agent]')).toBeNull()
    expect(state.$terminals.get().some(term => term.kind === 'user')).toBe(false)
    // Quietly surfaced tabs must still yield an active tab on a later manual open.
    act(() => state.setTerminalTakeover(true))
    expect(state.$terminals.get().find(term => term.id === state.$activeTerminalId.get())?.procId).toBe(
      `task-${preference}`
    )
    expect(mount.container!.querySelectorAll('[data-agent][data-active="true"]')).toHaveLength(1)
    expect(state.$terminals.get().some(term => term.kind === 'user')).toBe(false)
    mount.unmount()
    window.localStorage.clear()
    vi.resetModules()
  }
})
