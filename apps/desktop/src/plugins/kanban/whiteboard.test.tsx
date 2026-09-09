import type * as PluginSdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as KanbanApi from './api'
import { KanbanWhiteboard } from './whiteboard'

const mocks = vi.hoisted(() => ({
  notify: vi.fn(),
  saveWhiteboard: vi.fn(async (scene: KanbanApi.WhiteboardScene) => ({ scene, size: 1, updated_at: 1 }))
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof PluginSdk>()

  return {
    ...original,
    host: { ...original.host, notify: mocks.notify },
    useTheme: () => ({ renderedMode: 'light' })
  }
})

vi.mock('@excalidraw/excalidraw', () => ({
  Excalidraw: ({ onChange }: { onChange: (elements: readonly object[], appState: object, files: object) => void }) => (
    <button onClick={() => onChange([{ id: 'stroke-1', type: 'freedraw' }], {}, {})} type="button">
      Draw stroke
    </button>
  ),
  serializeAsJSON: (elements: readonly object[], appState: object, files: object) =>
    JSON.stringify({ elements, appState, files })
}))

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchWhiteboard: vi.fn(async () => ({
    scene: { elements: [], appState: {}, files: {} },
    updated_at: null
  })),
  saveWhiteboard: mocks.saveWhiteboard
}))

afterEach(() => {
  cleanup()
  mocks.notify.mockClear()
  mocks.saveWhiteboard.mockClear()
})

describe('Kanban whiteboard', () => {
  it('loads the board scene and autosaves a drawing change', async () => {
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <KanbanWhiteboard />
      </QueryClientProvider>
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Draw stroke' }))

    await waitFor(() => expect(mocks.saveWhiteboard).toHaveBeenCalledTimes(1), { timeout: 2_000 })
    expect(mocks.saveWhiteboard).toHaveBeenCalledWith({
      elements: [{ id: 'stroke-1', type: 'freedraw' }],
      appState: {},
      files: {}
    })
    expect(mocks.notify).not.toHaveBeenCalled()
  })
})
