import type { AssistantRuntime, ExternalStoreAdapter, ThreadMessage } from '@assistant-ui/react'
import { render, waitFor } from '@testing-library/react'
import { useEffect, useMemo } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { useIncrementalExternalStoreRuntime } from './incremental-external-store-runtime'

function RuntimeHarness({
  onRuntime,
  resetKey
}: {
  onRuntime: (runtime: AssistantRuntime) => void
  resetKey: string
}) {
  const store = useMemo<ExternalStoreAdapter<ThreadMessage>>(
    () => ({
      messages: [],
      isRunning: false,
      onNew: async () => {}
    }),
    []
  )

  const runtime = useIncrementalExternalStoreRuntime(store, { resetKey })

  useEffect(() => {
    onRuntime(runtime)
  }, [onRuntime, runtime])

  return null
}

describe('useIncrementalExternalStoreRuntime', () => {
  it('creates a fresh assistant runtime when the logical thread key changes', async () => {
    const runtimes: AssistantRuntime[] = []

    const onRuntime = vi.fn((runtime: AssistantRuntime) => {
      runtimes.push(runtime)
    })

    const { rerender } = render(<RuntimeHarness onRuntime={onRuntime} resetKey="session-a" />)

    await waitFor(() => expect(runtimes).toHaveLength(1))

    const firstRuntime = runtimes.at(-1)

    rerender(<RuntimeHarness onRuntime={onRuntime} resetKey="session-a" />)
    expect(runtimes.at(-1)).toBe(firstRuntime)

    rerender(<RuntimeHarness onRuntime={onRuntime} resetKey="session-b" />)

    await waitFor(() => expect(runtimes.at(-1)).not.toBe(firstRuntime))
  })
})
