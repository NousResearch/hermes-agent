import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { fetchProjectSessions, ProjectSessionsSuperseded } from '@/store/projects'

import type { SidebarProjectTree } from './projects/workspace-groups'
import { useEnteredProjectSessions } from './use-entered-project-sessions'

vi.mock('@/store/projects', () => ({
  fetchProjectSessions: vi.fn(),
  // The hook's guard is `instanceof`, so the test has to share this exact class.
  ProjectSessionsSuperseded: class ProjectSessionsSuperseded extends Error {}
}))
afterEach(cleanup)

// Long enough for the hook's bounded superseded retries (3 × 150 ms).
const RETRY_TIMEOUT_MS = 3000

// Stable identity: the hook re-runs its read whenever `treeRevision` changes,
// so an inline `[]` would restart the effect on every render.
const NO_REVISIONS: never[] = []

it('ignores departed drill-ins and clears failure on retry', async () => {
  let failOld!: (error: Error) => void
  vi.mocked(fetchProjectSessions)
    .mockImplementationOnce(
      () =>
        new Promise((_, reject) => {
          failOld = reject
        })
    )
    .mockResolvedValueOnce(null)
    .mockRejectedValueOnce(new Error('failed'))
    .mockResolvedValueOnce(null)
  const tree: never[] = []

  const { result, rerender } = renderHook(({ id }) => useEnteredProjectSessions(id, true, tree, 'default'), {
    initialProps: { id: 'old' }
  })

  rerender({ id: 'current' })
  await waitFor(() => expect(result.current.loading).toBe(false))
  await act(async () => failOld(new Error('late error')))
  expect(result.current.failed).toBe(false)
  act(() => result.current.retry())
  await waitFor(() => expect(result.current.failed).toBe(true))
  act(() => result.current.retry())
  await waitFor(() => expect(result.current.loading).toBe(false))
  expect(result.current.failed).toBe(false)
})

// A superseded read is not an answer: committing it painted the drill-in as an
// empty project (the overview node's lanes carry no rows), so the entered
// project showed branch headers with nothing under them.
it('retries a superseded read instead of committing it as an empty project', async () => {
  const hydrated = { id: 'current', repos: [], sessionCount: 2 } as unknown as SidebarProjectTree

  vi.mocked(fetchProjectSessions)
    .mockRejectedValueOnce(new ProjectSessionsSuperseded('superseded'))
    .mockRejectedValueOnce(new ProjectSessionsSuperseded('superseded'))
    .mockResolvedValue(hydrated)

  const { result } = renderHook(() => useEnteredProjectSessions('current', true, NO_REVISIONS, 'default'))

  await waitFor(() => expect(result.current.project).toBe(hydrated), { timeout: RETRY_TIMEOUT_MS })
  expect(result.current.failed).toBe(false)
  expect(result.current.loading).toBe(false)
})

it('reports failure — never an empty project — when the read stays superseded', async () => {
  vi.mocked(fetchProjectSessions).mockImplementation(() => Promise.reject(new ProjectSessionsSuperseded('superseded')))

  const { result } = renderHook(() => useEnteredProjectSessions('current', true, NO_REVISIONS, 'default'))

  await waitFor(() => expect(result.current.failed).toBe(true), { timeout: RETRY_TIMEOUT_MS })
  expect(result.current.project).toBe(null)
  expect(result.current.loading).toBe(false)
})
