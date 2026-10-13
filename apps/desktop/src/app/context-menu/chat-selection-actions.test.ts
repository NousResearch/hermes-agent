import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import { requestOneShot } from '@/lib/oneshot'

const mocks = vi.hoisted(() => ({
  owner: vi.fn(),
  speak: vi.fn(async () => true),
  translate: vi.fn(),
  ambient: vi.fn(),
  routed: vi.fn(async () => ({ text: 'translated' })),
  active: vi.fn(() => 'other-chat')
}))

vi.mock('@/store/session-states', () => ({ knownOwnerForSession: mocks.owner }))
vi.mock('@/store/session', () => ({ $activeSessionId: { get: mocks.active } }))
vi.mock('@/store/gateway', () => ({
  $gateway: { get: () => ({ request: mocks.ambient }) },
  requestGatewayForAgent: mocks.routed,
  requestGatewayForProfile: mocks.routed,
  retainGatewayForSessionTurn: vi.fn(() => () => {})
}))
vi.mock('@/store/selection-translate', () => ({ openSelectionTranslate: mocks.translate }))
vi.mock('@/lib/voice-playback', () => ({ playSpeechText: mocks.speak }))

import { captureChatSelection } from './chat-selection'
import { runChatSelectionAction } from './chat-selection-actions'

function selected() {
  document.body.innerHTML =
    '<div data-selection-session-id="source-chat"><div data-slot="aui_assistant-message-root">source words</div></div>'
  const message = document.querySelector('[data-slot]')!
  const range = document.createRange()
  range.selectNodeContents(message)
  window.getSelection()!.removeAllRanges()
  window.getSelection()!.addRange(range)

  return captureChatSelection(message)!
}

beforeEach(() => {
  vi.clearAllMocks()
  mocks.owner.mockReturnValue({
    connectionId: 'source-gateway',
    profile: 'desktop-profile',
    targetProfile: 'source-profile'
  })
  setApiRequestConnection('ambient-gateway')
  setApiRequestProfile('ambient-profile')
})
afterEach(() => {
  document.body.innerHTML = ''
  window.getSelection()?.removeAllRanges()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
})

it('uses the selected chat owner for speech and for later translation requests', async () => {
  const selection = selected()
  expect(await runChatSelectionAction('read-aloud', selection)).toBe(true)
  expect(mocks.speak).toHaveBeenCalledWith(
    'source words',
    expect.objectContaining({ connectionId: 'source-gateway', profile: 'source-profile', source: 'read-aloud' })
  )
  expect(await runChatSelectionAction('translate', selection)).toBe(true)
  const context = mocks.translate.mock.calls[0][1]
  expect(context.sessionId).toBe('source-chat')
  setApiRequestConnection('new-gateway')
  setApiRequestProfile('new-profile')
  await requestOneShot({ input: selection.text, sessionId: context.sessionId }, context.request)
  expect(mocks.ambient).not.toHaveBeenCalled()
  expect(mocks.routed).toHaveBeenCalled()
  const call = JSON.stringify(mocks.routed.mock.calls[0])
  expect(call).toContain('source-gateway')
  expect(call).toContain('source-chat')
  expect(call).not.toContain('new-gateway')
})

it('does nothing for a stale selection or an unknown secondary-chat owner', async () => {
  const selection = selected()
  mocks.owner.mockReturnValue(undefined)
  expect(await runChatSelectionAction('translate', selection)).toBe(false)
  expect(await runChatSelectionAction('read-aloud', selection)).toBe(false)
  window.getSelection()!.removeAllRanges()
  expect(await runChatSelectionAction('translate', selection)).toBe(false)
  expect(mocks.translate).not.toHaveBeenCalled()
  expect(mocks.speak).not.toHaveBeenCalled()
})
