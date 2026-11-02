import { expect, it } from 'vitest'

import { canonicalApprovalPreview } from './canonical-group-approval'
import type { CanonicalPendingAction } from './canonical-groups'

const pending = (approval: CanonicalPendingAction['approval'], request_id = 'r-1'): CanonicalPendingAction =>
  ({ kind: 'approval', member_id: 'ops', task_id: 'task', execution_generation: 1, request_id, approval })

const reviewable = (approval: CanonicalPendingAction['approval'], request_id?: string) =>
  canonicalApprovalPreview(pending(approval, request_id)).reviewable

it('offers a command for review only on its own exact request', () => {
  expect(canonicalApprovalPreview(pending({ command: ' rm -rf build ', description: 'Clear the build folder' }))).toEqual({
    actualCommand: 'rm -rf build', description: 'Clear the build folder', reviewable: true
  })
  expect(reviewable({ command: 'rm -rf build', request_id: 'r-1', prompt_id: 'r-1' })).toBe(true)
  expect(reviewable({ command: 'rm -rf build', request_id: 'r-2' })).toBe(false)
  expect(reviewable({ command: 'rm -rf build', prompt_id: 'r-2' })).toBe(false)
  expect(reviewable({ command: 'rm -rf build' }, ' ')).toBe(false)
  // No room approval on main carries an `edit`, and the preview ignores one, null included.
  expect(reviewable({ command: 'rm -rf build', edit: null } as CanonicalPendingAction['approval'])).toBe(true)
})

it('lets the description a plugin approval rule sends stand for its tool label, as the chat approval card does', () => {
  const plugin = { command: '<terminal> (plugin approval rule)', description: 'Approve running pwd' }
  expect(canonicalApprovalPreview(pending(plugin))).toEqual({ actualCommand: '', description: 'Approve running pwd', reviewable: true })
  expect(reviewable({ ...plugin, description: ' ' })).toBe(false)
  expect(reviewable({ ...plugin, prompt_id: 'r-2' })).toBe(false)
  expect(reviewable(plugin, '')).toBe(false)
})

it('authorizes nothing from a description alone or a bare tool label', () => {
  expect(reviewable({ description: 'Use a tool' })).toBe(false)
  expect(reviewable({ command: '<terminal> (shell)' })).toBe(false)
  expect(reviewable(undefined)).toBe(false)
})
