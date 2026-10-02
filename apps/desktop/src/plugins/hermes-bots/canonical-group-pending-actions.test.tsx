import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { CanonicalGroupPendingActions } from './canonical-group-pending-actions'
import { actCanonicalGroup } from './canonical-groups'
import type { CanonicalPendingAction } from './canonical-groups'

const { request } = vi.hoisted(() => ({ request: vi.fn(async (_route: unknown, _method: string, _params: unknown) => ({})) }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  return { ...sdk, host: { ...sdk.host, requestProfile: request } }
})
vi.mock('./canonical-group-labels', () => ({ useCanonicalGroupLabels: () => ({
  pendingApprovalTitle: '{name} needs your approval', approvalAction: 'Action', approvalCommand: 'Command',
  approvalChanges: 'Proposed changes', approvalBefore: 'Before', approvalAfter: 'After', approvalEmptyFile: 'Empty file',
  approvalDetailsMissing: 'Action details are unavailable. Check again before allowing this.',
  pendingRetryTitle: '{name} couldn’t start this reply.', pendingUnknownTitle: 'We couldn’t confirm whether {name} finished.',
  pendingFilesTitle: 'Sharing files from {name}…', pendingFilesCleanupTitle: 'Finishing file cleanup for {name}…',
  pendingFilesBlockedTitle: 'Files from {name} need attention.',
  pendingFilesBlockedHelp: 'File sharing could not finish. Check the computers running this group chat.',
  pendingStoppingTitle: 'Stopping {name}…', retryReply: 'Try again', skipReply: 'Skip this reply', pendingBot: 'Bot',
  pendingActionUnconfirmed: 'We couldn’t confirm this action. Refresh the group chat to check its status.',
  skipUnstartedWarning: 'This bot won’t reply to this message.', discardUnknown: 'Skip this reply?',
  discardWarning: 'The bot may have already acted. Skipping won’t undo its actions.', confirmDiscard: 'Skip reply',
  cancel: 'Cancel', deny: 'Don’t allow', allowOnce: 'Allow once', refresh: 'Refresh'
}) }))

afterEach(() => {cleanup(); request.mockClear()})
const members = [{ member_id: 'member-7', profile: 'default', handle: 'atlas', display_name: 'Atlas Bot' }]
const binding = { connectionId: 'original-computer', profile: 'default', roomId: 'original-room' }
const target = { member_id: 'member-7', task_id: 'original-task', execution_generation: 4 }
const noop = () => undefined

test('only reviewable supported approval details allow the exact room request; no normal-session or broad approval route', async () => {
  const action: CanonicalPendingAction = { ...target, kind: 'approval', request_id: 'approval-1', approval: {
    request_id: 'approval-1', command: 'rm -rf archive/draft-release-notes', description: 'Remove the old draft notes.',
    choices: ['once', 'deny', 'always', 'session']
  } }
  const onAction = async (pending: CanonicalPendingAction, choice?: 'once' | 'deny') => {await actCanonicalGroup(binding, pending, choice)}
  const view = render(<CanonicalGroupPendingActions actions={[action]} members={members} onAction={onAction} onDiscard={noop} onRefresh={noop} />)
  expect(screen.getByText('Atlas Bot needs your approval')).toBeTruthy()
  expect(screen.getByText(action.approval!.command!)).toBeTruthy()
  expect(screen.getByText(action.approval!.description!)).toBeTruthy()
  expect(screen.queryByText('member-7')).toBeNull()
  expect(screen.queryByRole('button', { name: /always|session|edit/i })).toBeNull()
  await act(async () => {fireEvent.click(screen.getByRole('button', { name: 'Allow once' }))})
  expect(request).toHaveBeenCalledOnce()
  expect(request.mock.calls[0]).toEqual([
    { connectionId: binding.connectionId, profile: 'default', targetProfile: 'default', mode: 'remote' },
    'groups.approve', { room_id: binding.roomId, ...target, request_id: 'approval-1', choice: 'once', profile: 'default' }
  ])

  const show = (approval: CanonicalPendingAction['approval']) => view.rerender(<CanonicalGroupPendingActions
    actions={[{ ...action, request_id: 'approval-2', approval }]} members={members} onAction={onAction} onDiscard={noop} onRefresh={noop} />)
  show({ command: '<terminal> (plugin approval rule)', choices: ['once', 'deny'] })
  expect(screen.getByText(/Action details are unavailable/)).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Allow once' })).toBeNull()
  expect(screen.getByRole('button', { name: 'Don’t allow' })).toBeTruthy()
  await act(async () => {fireEvent.click(screen.getByRole('button', { name: 'Don’t allow' }))})
  expect(request.mock.calls[1]).toEqual([
    { connectionId: binding.connectionId, profile: 'default', targetProfile: 'default', mode: 'remote' },
    'groups.approve', { room_id: binding.roomId, ...target, request_id: 'approval-2', choice: 'deny', profile: 'default' }
  ])
  show({ request_id: 'other-request', command: 'visible but wrong request', choices: ['once', 'deny'] })
  expect(screen.queryByRole('button', { name: 'Allow once' })).toBeNull()

  show({ edit: { tool_name: 'write_file', path: 'release-notes.md', old_text: 'Draft', new_text: 'Ready',
    arguments: { private_unrelated_field: 'must not render' } }, choices: ['once', 'deny'] })
  expect(screen.getByText('Proposed changes')).toBeTruthy()
  expect(screen.getByText('release-notes.md')).toBeTruthy()
  expect(screen.getByText('Draft')).toBeTruthy()
  expect(screen.getByText('Ready')).toBeTruthy()
  expect(screen.queryByText('must not render')).toBeNull()
  expect(screen.queryByRole('textbox')).toBeNull()
  expect(screen.getByRole('button', { name: 'Allow once' })).toBeTruthy()
  show({ command: 'Edit a file', edit: { opaque_private_payload: 'must not render' } })
  expect(screen.queryByRole('button', { name: 'Allow once' })).toBeNull()
})

test('unknown work never gains Retry and skip confirmation retains its original attempt through replacement and refusal', async () => {
  const unknown: CanonicalPendingAction = { ...target, kind: 'discard' }
  const onDiscard = vi.fn(async (action: CanonicalPendingAction) => {
    await actCanonicalGroup(binding, action)
    throw new Error('stale_generation')
  })
  const view = render(<CanonicalGroupPendingActions actions={[unknown]} members={members} onAction={noop} onDiscard={onDiscard} />)
  expect(screen.getByText('We couldn’t confirm whether Atlas Bot finished.')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Try again' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Skip this reply' }))
  expect(screen.getByText(/Skipping won’t undo/)).toBeTruthy()
  unknown.execution_generation = 5
  view.rerender(<CanonicalGroupPendingActions actions={[unknown]} members={members} onAction={noop} onDiscard={onDiscard} />)
  await act(async () => {fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Skip reply' }))})
  expect(onDiscard).toHaveBeenCalledWith({ ...target, kind: 'discard' })
  expect(request.mock.calls[0][1]).toBe('groups.discard')
  expect(request.mock.calls[0][2]).toMatchObject({ room_id: binding.roomId, execution_generation: 4, task_id: target.task_id })
  expect(screen.getByRole('dialog')).toBeTruthy()
  expect(screen.getByText(/We couldn’t confirm this action/)).toBeTruthy()
  expect(screen.queryByText('stale_generation')).toBeNull()
  fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Cancel' }))

  const retry: CanonicalPendingAction = { ...target, kind: 'retry' }
  view.rerender(<CanonicalGroupPendingActions actions={[retry, { ...retry, kind: 'discard' }]} members={members} onAction={noop} onDiscard={noop} />)
  expect(screen.getAllByTestId('group-chat-pending-action')).toHaveLength(1)
  expect(screen.getByText('Atlas Bot couldn’t start this reply.')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Skip this reply' }))
  expect(screen.getByText('This bot won’t reply to this message.')).toBeTruthy()
  expect(screen.queryByText(/Skipping won’t undo/)).toBeNull()
})


test('file publication and cleanup remain informational, including blocked refresh without execution controls', async () => {
  const onAction = vi.fn(), onDiscard = vi.fn(), onRefresh = vi.fn()
  const action: CanonicalPendingAction = { ...target, kind: 'output_retry', operation: 'ack', blocked: false }
  const view = render(<CanonicalGroupPendingActions actions={[action]} members={members}
    onAction={onAction} onDiscard={onDiscard} onRefresh={onRefresh} />)
  expect(screen.getByText('Sharing files from Atlas Bot…')).toBeTruthy()
  expect(screen.queryByRole('button')).toBeNull()
  view.rerender(<CanonicalGroupPendingActions actions={[{ ...action, operation: 'discard' }]} members={members}
    onAction={onAction} onDiscard={onDiscard} onRefresh={onRefresh} />)
  expect(screen.getByText('Finishing file cleanup for Atlas Bot…')).toBeTruthy()
  expect(screen.queryByRole('button')).toBeNull()
  view.rerender(<CanonicalGroupPendingActions actions={[{ ...action, blocked: true }]} members={members}
    onAction={onAction} onDiscard={onDiscard} onRefresh={onRefresh} />)
  expect(screen.getByText('Files from Atlas Bot need attention.')).toBeTruthy()
  expect(screen.queryByRole('button', { name: /Try again|Skip|Allow/i })).toBeNull()
  await act(async () => {fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))})
  expect(onRefresh).toHaveBeenCalledOnce()
  expect(onAction).not.toHaveBeenCalled()
  expect(onDiscard).not.toHaveBeenCalled()
  expect(request).not.toHaveBeenCalled()
})
