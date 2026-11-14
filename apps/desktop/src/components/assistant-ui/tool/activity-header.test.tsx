import { type ThreadMessage } from '@assistant-ui/react'
import {
  TOOL_ACTIVITY_HEADER_API_VERSION,
  TOOL_ACTIVITY_HEADER_AREA,
  type ToolActivityHeaderContext,
  type ToolActivityHeaderContribution
} from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useEffect, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createPluginContext } from '@/contrib/plugin'
import { $pluginDecisions, dropPlugin, setPluginEnabled } from '@/contrib/plugins-store'
import { registry } from '@/contrib/registry'
import { loadRuntimePlugin, unloadRuntimePlugin } from '@/contrib/runtime-loader'
import { $displayTimestamps } from '@/store/display-timestamps'
import { setShowToolActivityFromConfig } from '@/store/tool-activity'
import { clearDismissedToolRows } from '@/store/tool-dismiss'
import { $toolDisclosureStates } from '@/store/tool-view'

import { createdAt, stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '../test-utils'
import { Thread } from '../thread'

stubThreadEnvironment()
stubThreadViewportSize()

const disposers: (() => void)[] = []
const ctx = createPluginContext('header-fixture', dispose => disposers.push(dispose))
type Part = { toolName: string } & Partial<ToolActivityHeaderContext> & {
    args?: unknown
    result?: unknown
    timestamp?: number
  }

function message(parts: Part[], running = false): ThreadMessage {
  return {
    id: 'header-message',
    role: 'assistant',
    content: parts.map((part, index) => ({
      type: 'tool-call',
      toolCallId: `header-call-${index}`,
      args: {},
      argsText: '{}',
      ...part
    })),
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as unknown as ThreadMessage
}

function Transcript({ parts, running = false }: { parts: Part[]; running?: boolean }) {
  return (
    <ThreadRuntime messages={[message(parts, running)]}>
      <Thread />
    </ThreadRuntime>
  )
}

const read = { toolName: 'read_file', args: { path: '/repo/a.txt' }, result: { content: 'full native output' } }

function register(renderHeader: ToolActivityHeaderContribution['render'], id = 'decoration', order?: number) {
  return ctx.register({
    id,
    area: TOOL_ACTIVITY_HEADER_AREA,
    order,
    data: {
      apiVersion: TOOL_ACTIVITY_HEADER_API_VERSION,
      render: renderHeader
    } satisfies ToolActivityHeaderContribution
  })
}

beforeEach(() => {
  $displayTimestamps.set(false)
  $toolDisclosureStates.set({})
  clearDismissedToolRows()
  setShowToolActivityFromConfig(undefined)
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  unloadRuntimePlugin('header-runtime-fixture')
  dropPlugin('header-runtime-fixture')
  $pluginDecisions.set({})
  $toolDisclosureStates.set({})
  clearDismissedToolRows()
  setShowToolActivityFromConfig(undefined)
  vi.restoreAllMocks()
})

describe('ordinary tool header SDK contributions', () => {
  it('keeps the exact native header DOM when absent, declined or unregistered', () => {
    const { container } = render(<Transcript parts={[read]} />)
    const native = container.querySelector('[data-tool-row]')!.outerHTML

    let dispose = () => {}

    act(() => {
      dispose = register(() => null)
    })
    expect(container.querySelector('[data-tool-row]')!.outerHTML).toBe(native)
    act(() => dispose())
    act(() => {
      dispose = register(() => <span>inline annotation</span>)
    })
    expect(screen.getByText('inline annotation').closest('[data-tool-row]')).not.toBeNull()
    act(() => dispose())
    expect(container.querySelector('[data-tool-row]')!.outerHTML).toBe(native)
  })

  it('adds content inside the existing disclosure button without replacing native output', () => {
    register(({ toolName }) => <span data-testid="tool-decoration">{toolName} annotation</span>)
    const { container } = render(<Transcript parts={[read]} />)
    const decoration = screen.getByTestId('tool-decoration')
    const button = decoration.closest('button')!

    expect(button.getAttribute('aria-expanded')).toBe('false')
    expect(button.textContent).toContain('Read')
    expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(1)
    fireEvent.click(button)
    expect(button.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByText('full native output')).toBeTruthy()
    fireEvent.click(button)
    expect(button.getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByText('full native output')).toBeNull()
  })

  it('updates host state without remounting a stateful decoration and cleans it up', async () => {
    const contexts: ToolActivityHeaderContext[] = []
    const unmount = vi.fn()
    let mounts = 0

    function Annotation(context: ToolActivityHeaderContext) {
      const [instance] = useState(() => ++mounts)
      useEffect(() => unmount, [])
      contexts.push(context)

      return (
        <span data-testid="stateful">
          {instance}:{context.status}:{String(context.expanded)}
        </span>
      )
    }

    const dispose = register(Annotation)
    const { rerender } = render(<Transcript parts={[{ toolName: 'terminal', timestamp: 100 }]} running />)
    expect(screen.getByTestId('stateful').textContent).toBe('1:running:false')
    rerender(
      <Transcript
        parts={[{ toolName: 'terminal', timestamp: 100, completedAt: 103, result: { stdout: 'done', exit_code: 0 } }]}
      />
    )
    await waitFor(() => expect(screen.getByTestId('stateful').textContent).toBe('1:success:false'))
    expect(contexts.at(-1)).toMatchObject({
      startedAt: 100,
      completedAt: 103,
      exitCode: 0,
      hasResult: true,
      hasDetails: true
    })
    fireEvent.click(screen.getByTestId('stateful').closest('button')!)
    expect(screen.getByTestId('stateful').textContent).toBe('1:success:true')
    expect(mounts).toBe(1)
    act(() => dispose())
    expect(unmount).toHaveBeenCalledTimes(1)
  })

  it('provides a frozen scalar-only context and leaves missing history fields undefined', () => {
    let context: ToolActivityHeaderContext | undefined
    register(value => {
      context = value

      return null
    })
    render(<Transcript parts={[read]} />)
    expect(Object.isFrozen(context)).toBe(true)
    expect(context).toMatchObject({
      toolName: 'read_file',
      status: 'success',
      hasResult: true,
      hasDetails: true,
      expanded: false
    })

    for (const key of ['startedAt', 'completedAt', 'exitCode', 'innerToolName', 'interrupted']) {
      expect(context?.[key as keyof ToolActivityHeaderContext]).toBeUndefined()
    }

    expect(context).not.toHaveProperty('args')
    expect(context).not.toHaveProperty('result')
    expect(context).not.toHaveProperty('toolResultMetadata')
    expect(context).not.toHaveProperty('progress')
    expect(Reflect.set(context!, 'status', 'error')).toBe(false)
  })

  it.each([
    { interrupted: undefined, status: 'warning' },
    { interrupted: true, status: 'notice' }
  ])(
    'does not invent completion timing or success for a settled missing result ($status)',
    ({ interrupted, status }) => {
      let context: ToolActivityHeaderContext | undefined
      register(value => {
        context = value

        return null
      })
      render(<Transcript parts={[{ toolName: 'terminal', interrupted }]} />)
      expect(context).toMatchObject({ status, hasResult: false, interrupted })
      expect(context?.startedAt).toBeUndefined()
      expect(context?.completedAt).toBeUndefined()
      expect(context?.exitCode).toBeUndefined()
    }
  )

  it('reports host failure and bridge metadata without deriving status from exit code', async () => {
    const contexts: ToolActivityHeaderContext[] = []
    register(context => {
      contexts.push(context)

      return null
    })

    const { rerender } = render(
      <Transcript parts={[{ toolName: 'terminal', result: { error: 'permission denied' } }]} />
    )

    expect(contexts.at(-1)?.status).toBe('error')
    rerender(
      <Transcript
        parts={[{ toolName: 'terminal', innerToolName: 'subcommand', result: { stdout: 'no matches', exit_code: 1 } }]}
      />
    )
    await waitFor(() =>
      expect(contexts.at(-1)).toMatchObject({ status: 'success', innerToolName: 'subcommand', exitCode: 1 })
    )
  })

  it('composes every registration in order, breaks ties by registration and replaces the same id', () => {
    register(() => <span data-decoration="">first tie</span>, 'first', 5)
    register(() => <span data-decoration="">second tie</span>, 'second', 5)
    register(() => <span data-decoration="">priority</span>, 'priority', -1)
    const { container } = render(<Transcript parts={[read]} />)
    const labels = () => Array.from(container.querySelectorAll('[data-decoration]'), node => node.textContent)
    expect(labels()).toEqual(['priority', 'first tie', 'second tie'])
    act(() => {
      register(() => <span data-decoration="">replacement</span>, 'first', 5)
    })
    expect(labels()).toEqual(['priority', 'second tie', 'replacement'])
    expect(registry.getArea(TOOL_ACTIVITY_HEADER_AREA)).toHaveLength(3)
  })

  it('ignores incompatible, missing and malformed payloads', () => {
    for (const [id, data] of [
      ['future', { apiVersion: 2, render: () => <span>not compatible</span> }],
      ['missing', undefined],
      ['invalid', { apiVersion: 1, render: 'not a function' }]
    ] as const) {
      ctx.register({ id, area: TOOL_ACTIVITY_HEADER_AREA, data })
    }

    const { container } = render(<Transcript parts={[read]} />)
    expect(screen.queryByText('not compatible')).toBeNull()
    expect(container.querySelector('[data-tool-row]')).not.toBeNull()
    expect(container.querySelector('[data-tool-row] button[aria-expanded="false"]')).not.toBeNull()
  })

  it('contains render errors, keeps other decorators and native disclosure, and retries only on replacement', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})

    const broken = vi.fn(() => {
      throw new Error('fixture failure')
    })

    register(broken)
    register(() => <span>healthy decorator</span>, 'healthy')
    const { container, rerender } = render(<Transcript parts={[read]} />)
    expect(broken).toHaveBeenCalled()
    const attempts = broken.mock.calls.length
    expect(screen.getByText('healthy decorator')).toBeTruthy()
    fireEvent.click(container.querySelector('[data-tool-row] button[aria-expanded="false"]')!)
    expect(screen.getByText('full native output')).toBeTruthy()
    rerender(<Transcript parts={[read]} />)
    expect(broken).toHaveBeenCalledTimes(attempts)
    act(() => {
      register(() => <span>recovered decorator</span>)
    })
    expect(screen.getByText('recovered decorator')).toBeTruthy()
    expect(screen.getByText('full native output')).toBeTruthy()
  })

  it('contains commit-lifecycle errors and untyped payload accessors', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    register(function BadEffect() {
      useEffect(() => {
        throw new Error('commit fixture')
      }, [])

      return <span>bad effect</span>
    })
    ctx.register({
      id: 'accessor',
      area: TOOL_ACTIVITY_HEADER_AREA,
      data: {
        get apiVersion() {
          throw new Error('getter fixture')
        }
      }
    })
    const { container } = render(<Transcript parts={[read]} />)
    expect(screen.queryByText('bad effect')).toBeNull()
    fireEvent.click(container.querySelector('[data-tool-row] button[aria-expanded="false"]')!)
    expect(screen.getByText('full native output')).toBeTruthy()
  })

  it('keeps native run grouping and decorates only when the user reveals its rows', async () => {
    const callback = vi.fn(({ toolName }: ToolActivityHeaderContext) => (
      <span data-decoration="">{toolName} decoration</span>
    ))

    register(callback)
    const { container } = render(<Transcript parts={[read, { ...read, args: { path: '/repo/b.txt' } }]} />)
    const summary = await screen.findByText(/Explored 2 files/)
    expect(callback).not.toHaveBeenCalled()
    expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(0)
    fireEvent.click(summary.closest('button')!)
    expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(2)
    expect(container.querySelectorAll('[data-decoration]')).toHaveLength(2)
    fireEvent.click(summary.closest('button')!)
    expect(container.querySelectorAll('[data-decoration]')).toHaveLength(0)
  })

  it('does not decorate specialized file-edit cards or reveal hidden activity', () => {
    const callback = vi.fn(() => <span>unexpected decoration</span>)
    register(callback)
    setShowToolActivityFromConfig('off')

    const { container } = render(
      <Transcript
        parts={[
          read,
          { toolName: 'patch', args: { path: '/repo/a.txt' }, result: { inline_diff: '--- a\n+++ b\n-old\n+new' } }
        ]}
      />
    )

    expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(1)
    expect(callback).not.toHaveBeenCalled()
    expect(screen.queryByText('unexpected decoration')).toBeNull()
  })

  it('uses the real runtime SDK loader and restores native behavior on disable, re-enable and unload', async () => {
    const originalBlob = globalThis.Blob
    const fixtureGlobals = globalThis as unknown as Record<string, unknown>
    fixtureGlobals.__toolHeaderRender = () => <span>runtime decoration</span>
    vi.stubGlobal(
      'Blob',
      class {
        constructor(public parts: string[]) {}
      }
    )

    const objectUrl = vi
      .spyOn(URL, 'createObjectURL')
      .mockImplementation(
        blob =>
          `data:text/javascript;base64,${Buffer.from((blob as unknown as { parts: string[] }).parts.join('')).toString('base64')}`
      )

    const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})

    try {
      const { container } = render(<Transcript parts={[read]} />)
      const native = container.querySelector('[data-tool-row]')!.outerHTML
      let loaded: string | null = null
      await act(async () => {
        loaded = await loadRuntimePlugin(
          `
          import { TOOL_ACTIVITY_HEADER_AREA, TOOL_ACTIVITY_HEADER_API_VERSION } from '@hermes/plugin-sdk'
          export default {
            id: 'header-runtime-fixture',
            register(ctx) { ctx.register({ id: 'label', area: TOOL_ACTIVITY_HEADER_AREA,
              data: { apiVersion: TOOL_ACTIVITY_HEADER_API_VERSION, render: globalThis.__toolHeaderRender } }) }
          }
        `,
          'header-runtime-fixture'
        )
      })
      expect(loaded).toBe('header-runtime-fixture')
      expect(screen.getByText('runtime decoration')).toBeTruthy()
      await act(() => setPluginEnabled('header-runtime-fixture', false))
      expect(container.querySelector('[data-tool-row]')!.outerHTML).toBe(native)
      await act(() => setPluginEnabled('header-runtime-fixture', true))
      expect(screen.getByText('runtime decoration')).toBeTruthy()
      act(() => unloadRuntimePlugin('header-runtime-fixture'))
      await waitFor(() => expect(container.querySelector('[data-tool-row]')!.outerHTML).toBe(native))
    } finally {
      globalThis.Blob = originalBlob
      delete fixtureGlobals.__toolHeaderRender
      objectUrl.mockRestore()
      revokeUrl.mockRestore()
    }
  })
})
