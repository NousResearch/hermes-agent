import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $activeBucket,
  $activeTab,
  $contextData,
  $contextInspectorOpen,
  $contextSource,
  closeContextInspector
} from '@/store/context-inspector'
import type { ContextFull } from '@/types/hermes'

import { ContextInspectorView } from './context-inspector'

const payload = (overrides: Partial<ContextFull> = {}): ContextFull => ({
  available: true,
  context_max: 200_000,
  context_used: 12_345,
  exact_capture_available: false,
  messages: [
    {
      content_text: '<script>alert(1)</script>\nSECRET_TOKEN=abc',
      index: 0,
      raw: { role: 'system' },
      role: 'system',
      tokens: 100
    },
    {
      content_text: '<img src=x onerror=alert(1)>\nSECRET_TOKEN=abc',
      index: 1,
      raw: { role: 'user' },
      role: 'user',
      tokens: 20
    },
    {
      content_text: 'Assistant response',
      index: 2,
      raw: { reasoning: 'thoughts', role: 'assistant', tool_calls: [{ function: { name: 'read_file' }, id: 'call-1' }] },
      role: 'assistant',
      tokens: 40
    },
    {
      content_text: '{"content":"tool result"}',
      index: 3,
      raw: { name: 'read_file', role: 'tool', tool_call_id: 'call-1' },
      role: 'tool',
      tokens: 30
    }
  ],
  model: 'gpt-test',
  raw_unredacted: true,
  slices: [
    {
      bucket: 'system',
      content_text: '<img src=x onerror=alert(1)>\nSECRET_TOKEN=abc',
      id: 'system_prompt',
      label: 'System prompt',
      source_accuracy: 'cached_exact',
      tokens: 100,
      truncated: true
    },
    {
      bucket: 'tools',
      content_text: 'Tool schemas',
      id: 'tool_definitions',
      label: 'Tool definitions',
      source_accuracy: 'reconstructed_current',
      tokens: 60,
      truncated: false
    }
  ],
  source: 'reconstructed_base',
  source_label: 'Reconstructed base',
  state: 'ready',
  ...overrides
})

function renderReady(data = payload()) {
  $contextInspectorOpen.set(true)
  $contextData.set(data)
  $contextSource.set({ runtimeSessionId: 'runtime-1', sessionId: 'runtime-1', status: 'ready' })
  render(<ContextInspectorView onClose={() => undefined} />)
}

describe('ContextInspectorView', () => {
  beforeEach(() => {
    closeContextInspector()
    $contextData.set(null)
    $contextSource.set({ status: 'idle' })
    $activeBucket.set('all')
    $activeTab.set('transcript')
  })

  afterEach(() => cleanup())

  it('renders raw content as React text without creating HTML nodes or redacting visible secrets', () => {
    renderReady()

    const pre = screen.getByText(
      (content: string, element: Element | null) => element?.tagName === 'PRE' && content.includes('<img src=x onerror=alert(1)>')
    )

    expect(pre.textContent).toContain('<img src=x onerror=alert(1)>')
    expect(pre.textContent).toContain('SECRET_TOKEN=abc')
    expect(pre.textContent).not.toContain('&lt;img')
    expect(document.querySelector('img')).toBeNull()
    expect(document.querySelector('script')).toBeNull()
  })

  it('defaults to the ordered transcript and filters by bucket chips', () => {
    renderReady()

    expect(screen.getByRole('tab', { name: /ordered transcript/i }).getAttribute('aria-selected')).toBe('true')
    expect(screen.getByText(/Assistant response/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /tools/i }))

    expect(screen.getByText(/tool_call_id: call-1/i)).toBeTruthy()
    expect(screen.queryByText(/Assistant response/)).toBeNull()
  })

  it('switches to the Advanced / Raw JSON tab with a tree and pretty payload pane', () => {
    renderReady()

    fireEvent.click(screen.getByRole('tab', { name: /advanced \/ raw json/i }))

    expect(screen.getByText('slices')).toBeTruthy()
    expect(screen.getByText('messages [4]')).toBeTruthy()

    const rawJson = screen.getByText(
      (content: string, element: Element | null) => element?.tagName === 'PRE' && content.includes('"source_label": "Reconstructed base"')
    )

    expect(rawJson.textContent).toContain('SECRET_TOKEN=abc')
  })

  it('shows empty, loading, error, source-help, and truncation states', () => {
    $contextInspectorOpen.set(true)
    $contextData.set(null)
    $contextSource.set({ status: 'loading', sessionId: 'runtime-1', runtimeSessionId: 'runtime-1' })
    const { rerender } = render(<ContextInspectorView onClose={() => undefined} />)
    expect(screen.getByText(/Loading full context/i)).toBeTruthy()

    $contextSource.set({ error: 'gateway offline', status: 'error', sessionId: 'runtime-1', runtimeSessionId: 'runtime-1' })
    rerender(<ContextInspectorView onClose={() => undefined} />)
    expect(screen.getByText(/Context failed to load: gateway offline/i)).toBeTruthy()

    $contextData.set(payload({ available: false, messages: [], slices: [], state: 'agent_not_built' }))
    $contextSource.set({ status: 'empty', sessionId: 'runtime-1', runtimeSessionId: 'runtime-1' })
    rerender(<ContextInspectorView onClose={() => undefined} />)
    expect(screen.getByText(/available after the agent initializes/i)).toBeTruthy()

    $contextData.set(payload())
    $contextSource.set({ status: 'ready', sessionId: 'runtime-1', runtimeSessionId: 'runtime-1' })
    rerender(<ContextInspectorView onClose={() => undefined} />)
    expect(screen.getByText(/excludes per-turn ephemeral injections/i)).toBeTruthy()
    expect(screen.getByText(/truncated by the backend/i)).toBeTruthy()
  })
})
