import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $backgroundStatusBySession, reconcileBackgroundProcesses } from '@/store/composer-status'

import {
  registerAgentTerminalWriter,
  syncAgentTerminalSnapshot,
  writeAgentTerminalChunk
} from './agent-terminal-stream'
import { $activeTerminalId, $terminals, closeTerminal, openAgentTerminal } from './terminals'
import { useAgentTerminal } from './use-agent-terminal'
import { TerminalWorkspace } from './workspace'

const screens = vi.hoisted(() => new Map<string, string[]>())

// Keep the store, workspace, attachment hook and stream real; only the xterm
// renderer and instance chrome are replaced so each tab's writes are observable.
vi.mock('@xterm/xterm', () => ({
  Terminal: class {
    readonly rows = 24
    readonly unicode = { activeVersion: '6' }
    readonly writes: string[] = []
    options: Record<string, unknown>

    constructor(options: Record<string, unknown>) {
      this.options = { ...options }
    }

    open(host: HTMLElement) {
      screens.set(host.dataset.procId!, this.writes)
    }

    write(chunk: string) {
      this.writes.push(chunk)
    }

    attachCustomKeyEventHandler = vi.fn()
    clearSelection = vi.fn()
    dispose = vi.fn()
    focus = vi.fn()
    getSelection = vi.fn(() => '')
    loadAddon = vi.fn()
    onSelectionChange = vi.fn(() => ({ dispose: vi.fn() }))
    refresh = vi.fn()
    selectAll = vi.fn()
  }
}))

vi.mock('@xterm/addon-fit', () => ({
  FitAddon: class {
    fit = vi.fn()
  }
}))

vi.mock('@xterm/addon-unicode11', () => ({ Unicode11Addon: class {} }))
vi.mock('@xterm/addon-web-links', () => ({ WebLinksAddon: class {} }))
vi.mock('@xterm/addon-webgl', () => ({
  WebglAddon: class {
    clearTextureAtlas = vi.fn()
    dispose = vi.fn()
    onContextLoss = vi.fn()
  }
}))

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ renderedMode: 'dark', theme: { terminal: {} }, themeName: 'test' })
}))

vi.mock('./instance', () => ({
  AgentTerminalInstance: (props: { active: boolean; id: string; procId: string }) => {
    const { hostRef } = useAgentTerminal(props)

    return <div data-proc-id={props.procId} ref={hostRef} />
  },
  TerminalInstance: () => null
}))

afterEach(() => {
  cleanup()
  $terminals.set([])
  $activeTerminalId.set(null)
  $backgroundStatusBySession.set({})
  screens.clear()
  Reflect.deleteProperty(document, 'fonts')
})

describe('quiet agent terminal recovery', () => {
  it('reopens an evicted tab from an unchanged registry exactly once while other writers remain mounted', async () => {
    Object.defineProperty(document, 'fonts', {
      configurable: true,
      value: { load: vi.fn().mockResolvedValue([]) }
    })
    const procs = Array.from({ length: 26 }, (_, index) => ({
      command: `quiet-command-${index}`,
      output_tail: `snapshot-marker-${index}\r\n`,
      session_id: `quiet-${index}`,
      status: 'running'
    }))
    reconcileBackgroundProcesses('other-session', procs.slice(0, 25))
    reconcileBackgroundProcesses('target-session', procs.slice(25))
    const background = $backgroundStatusBySession.get()
    const target = procs[25]!
    const expected = `$ ${target.command}\r\n${target.output_tail}`

    render(<TerminalWorkspace onAddSelectionToChat={() => {}} />)
    await waitFor(() => expect(screens.size).toBe(26))
    const original = screens.get(target.session_id)!
    expect(original.join('')).toBe(expected)
    const neighbor = screens.get(procs[24]!.session_id)!
    const neighborOutput = neighbor.join('')
    const tab = $terminals.get().find(entry => entry.procId === target.session_id)!

    act(() => closeTerminal(tab.id))
    expect($terminals.get()).toHaveLength(25)
    // The remaining mounted writers already exceed the cache's 24-process
    // budget: closing must really discard this tab, not grant a cache exemption.
    const replay = vi.fn()
    const stopProbe = registerAgentTerminalWriter(target.session_id, replay)
    expect(replay).not.toHaveBeenCalled()
    stopProbe()

    act(() => openAgentTerminal(target.session_id, target.command))
    await waitFor(() => expect(screens.get(target.session_id)).not.toBe(original))
    const recovered = screens.get(target.session_id)!
    expect($backgroundStatusBySession.get()).toBe(background)
    expect(recovered.join('')).toBe(expected)
    expect(neighbor.join('')).toBe(neighborOutput)
    // A delayed cleanup from the old attachment must not detach its replacement.
    stopProbe()

    // Both an unchanged registry snapshot and a later snapshot containing live
    // bytes must respect the restored fence rather than duplicate either source.
    syncAgentTerminalSnapshot(target.session_id, target.output_tail)
    const live = 'live-after-reopen\r\n'
    writeAgentTerminalChunk(target.session_id, live)
    syncAgentTerminalSnapshot(target.session_id, target.output_tail + live)
    expect(recovered.join('')).toBe(expected + live)
    expect(neighbor.join('')).toBe(neighborOutput)

    act(() => closeTerminal($activeTerminalId.get()!))
    const coldReplay = vi.fn()
    const stopColdProbe = registerAgentTerminalWriter(target.session_id, coldReplay)
    expect(coldReplay).not.toHaveBeenCalled()
    stopColdProbe()
  })
})
