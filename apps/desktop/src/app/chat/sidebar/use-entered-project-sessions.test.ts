import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { fetchProjectSessions } from '@/store/projects'

import type { SidebarProjectTree } from './projects/workspace-groups'
import { useEnteredProjectSessions } from './use-entered-project-sessions'

vi.mock('@/store/projects', () => ({ fetchProjectSessions: vi.fn() }))
afterEach(cleanup)

const node = (id: string, sessionCount = 0): SidebarProjectTree => ({
  id,
  label: id,
  path: `/${id}`,
  repos: [],
  sessionCount
})

// #77591 D: over a remote gateway every `projects.project_sessions` call is a
// full hydrated tree build, and the tree refreshes on every sessions.changed
// and window focus. Only a change to the entered project may refetch it.
it('refetches only when the entered project changes, and keeps an unchanged result', async () => {
  const fetched = vi.mocked(fetchProjectSessions)
  fetched.mockReset()
  fetched.mockImplementation(async id => ({ ...node(id, 1), repos: [] }))

  // `$projectTree` keeps unchanged nodes by reference across refreshes, so a
  // refresh that only touched another project (or nothing at all) is a new
  // array holding the same entered node.
  const enteredNode = node('a', 1)

  const { result, rerender } = renderHook(({ tree }) => useEnteredProjectSessions('a', true, tree, 'default'), {
    initialProps: { tree: [enteredNode, node('b')] }
  })

  await waitFor(() => expect(result.current.project?.id).toBe('a'))
  const first = result.current.project

  rerender({ tree: [enteredNode, node('b', 5)] })
  rerender({ tree: [enteredNode, node('b', 5)] })
  await act(async () => {})
  expect(fetched).toHaveBeenCalledTimes(1)

  // The entered project itself changed: refetch in the background, with no
  // skeleton, and keep the same object when the answer is unchanged.
  rerender({ tree: [node('a', 2), node('b', 5)] })
  expect(result.current.loading).toBe(false)
  await waitFor(() => expect(fetched).toHaveBeenCalledTimes(2))
  await act(async () => {})
  expect(result.current.project).toBe(first)
})

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
