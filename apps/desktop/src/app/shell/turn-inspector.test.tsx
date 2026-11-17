import type { SessionRenderSnapshotResult } from '@hermes/shared'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { TurnInspector } from './turn-inspector'

const snapshot: SessionRenderSnapshotResult = {
  mcp_servers: ['gh'],
  model: 'model-b',
  provider: 'nous',
  segments: [
    { changes: [], model: 'model-a', prompt_chars: 40, provider: 'nous', session_id: 'seg-1', tool_count: 2 },
    {
      changes: [
        { after: 'model-b', before: 'model-a', kind: 'model_changed' },
        { after: 'h2', before: 'h1', kind: 'prompt_changed' },
        { kind: 'tool_added', name: 'mcp__gh__issues' },
        { kind: 'tool_removed', name: 'read_file' }
      ],
      model: 'model-b',
      prompt_chars: 26,
      provider: 'nous',
      session_id: 'seg-2',
      tool_count: 2
    }
  ],
  session_id: 'seg-2',
  system_prompt: 'You are Hermes, compacted.',
  tools: [{ description: 'Run shell commands', name: 'terminal', toolset: 'terminal' }],
  tools_source: 'live',
  toolsets: ['terminal']
}

afterEach(() => {
  cleanup()
})

describe('TurnInspector', () => {
  it('reads the stored snapshot and renders one chip per lineage change', async () => {
    const requestGateway = vi.fn().mockResolvedValue(snapshot)

    render(<TurnInspector requestGateway={requestGateway} sessionId="runtime-1" />)

    await waitFor(() => expect(screen.getByText('Tool added: mcp__gh__issues')).toBeTruthy())
    expect(requestGateway).toHaveBeenCalledWith('session.render_snapshot', { session_id: 'runtime-1' })
    expect(screen.getByText('Model changed: model-a → model-b')).toBeTruthy()
    expect(screen.getByText('Prompt changed')).toBeTruthy()
    expect(screen.getByText('Tool removed: read_file')).toBeTruthy()
    expect(screen.getAllByText('Run shell commands').length).toBeGreaterThan(0)
    expect(screen.getByText('gh')).toBeTruthy()
  })
})
