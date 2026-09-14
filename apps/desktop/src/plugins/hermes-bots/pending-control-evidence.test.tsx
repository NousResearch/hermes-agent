import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { actCanonicalGroup } from './canonical-groups'
import { $groupClarify } from './group-chat'
import { GroupClarifyCard } from './group-chat-parts'
import { answerGroupClarify } from './group-turns'
import { syncHostedRoomApprovals } from './hosted-room-approval-state'
import { translateBots } from './i18n-test-helper'

const request = vi.hoisted(() => vi.fn())
const unavailable = 'Controls are unavailable here. This is a read-only observation from the execution owner.'
vi.mock('@hermes/plugin-sdk', async original => {
  const actual = await original<typeof HermesSdk>()
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')

  return { ...actual, ...await pluginSdkMock(gateway.host),
    useI18n: () => ({ locale: 'en', t: en }), usePluginI18n: () => translateBots,
    host: { ...gateway.host, requestProfile: request, request } }
})

const action = {
  kind: 'approval', member_id: 'member', task_id: 'task', execution_generation: 3, request_id: 'prompt',
  control_supported: false, admission_id: 'admission', target_execution_generation: 17,
  approval: { command: 'fixture', description: 'Observed prompt', choices: [], control_supported: false,
    admission_id: 'admission', target_execution_generation: 17 }
}

const binding = { connectionId: 'local', profile: 'default', roomId: 'room' }
const original = window.hermesDesktop

beforeEach(() => {
  request.mockReset()
  localStorage.clear()
  $groupClarify.set({})
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined })
})
afterEach(() => {
  cleanup()
  $groupClarify.set({})
  window.hermesDesktop = original
  vi.restoreAllMocks()
})

it('refuses unsupported canonical actions before any request', async () => {
  await expect(actCanonicalGroup(binding, action, 'once')).rejects.toThrow('unavailable')
  expect(request).not.toHaveBeenCalled()
})

it('retains unsupported hosted evidence without a decision card or cached old choices', async () => {
  const members = [{ name: 'ops', title: '' }]
  const room = { room_id: 'room', members: [{ member_id: 'member' }] }
  syncHostedRoomApprovals('Room', room, members, [{ ...action, control_supported: true,
    approval: { description: 'Observed prompt', command: 'fixture', choices: ['once', 'deny'] } }])
  const prior = Object.values($groupClarify.get())[0]
  expect(prior.choices).toEqual(['once', 'deny'])
  syncHostedRoomApprovals('Room', room, members, [action])
  const observed = Object.values($groupClarify.get())[0]
  expect(observed).not.toBe(prior)
  expect(observed).toMatchObject({ controlSupported: false, choices: [],
    hostedApproval: { admissionId: 'admission', targetExecutionGeneration: 17, executionGeneration: 3 } })
  render(<GroupClarifyCard entry={observed} members={members} />)
  expect(screen.getByText('Observed prompt')).toBeTruthy()
  expect(screen.getByText(unavailable)).toBeTruthy()
  expect(screen.queryAllByRole('button')).toHaveLength(0)
  await expect(answerGroupClarify(observed, members[0], 'once')).rejects.toThrow('unavailable')
  expect(request).not.toHaveBeenCalled()
})

it('canonical pending observations do not expose approval buttons', async () => {
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {
      return { room: { room_id: 'room', name: 'Room' }, driver_status: { pending_actions: [action] } }
    }

    if (method === 'groups.log') {
      return { events: [] }
    }

    throw new Error(`Unexpected request: ${method}`)
  })
  await act(async () => { render(<CanonicalGroupWorkspace binding={binding} />) })
  expect(await screen.findByText(unavailable)).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Allow once' })).toBeNull()
  expect(screen.queryByRole('button', { name: 'Deny' })).toBeNull()
  expect(request.mock.calls.every(([, method]) => ['groups.state', 'groups.log'].includes(method))).toBe(true)
})
