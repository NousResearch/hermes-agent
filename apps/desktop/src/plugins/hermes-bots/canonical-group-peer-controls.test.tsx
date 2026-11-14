import type * as HermesSdk from '@hermes/plugin-sdk'
// eslint-disable-next-line no-restricted-imports -- Exercise the same serialized errors as the shared client.
import { jsonRpcErrorFromFrame } from '@hermes/shared'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { answerGroupRequests, CANONICAL_GROUP_CAPABILITIES, DISBAND_FAILURES, localGatewaySdkMock } from './canonical-group-contract-test-utils'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { $canonicalGroupBindings, registerCanonicalGroup } from './canonical-group-registry'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { disbandCanonicalGroup } from './canonical-groups'
import wire from './fixtures/peer-control-responses.json'
import { $groupChats, updateGroupChat } from './group-chat'
import { translateBotsIn } from './i18n-test-helper'

const { request, locale } = vi.hoisted(() => ({ request: vi.fn(), locale: { value: 'en' } }))
vi.mock('@hermes/plugin-sdk', async importOriginal =>
  localGatewaySdkMock(await importOriginal<typeof HermesSdk>(), request, () => translateBotsIn(locale.value)))

// Captured through AuthorityConnection.dispatch at gateway 62f2b28d15. The fixture generator
// verifies the durable receipt is absent/present; neither state exposes that receipt to Desktop.
// Typed Disband reasons are injected at the service boundary, then serialized by the dispatcher.
// The untyped frame is ws.py's dispatch-crash reply on the current gateway.
const frame = (name: keyof typeof wire) => JSON.parse(wire[name])
const binding = { connectionId: 'local', profile: 'default', roomId: 'linked' }
const locales = Object.keys(CANONICAL_GROUP_LOCALES) as (keyof typeof CANONICAL_GROUP_LOCALES)[]
const originalDesktop = window.hermesDesktop


beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined })
  $canonicalGroupBindings.set({})
  $groupChats.set({})
})
afterEach(() => {
  cleanup()
  request.mockReset()
  localStorage.clear()
  window.hermesDesktop = originalDesktop
})

function gateway(state: () => unknown, control: (method: string) => unknown, log = () => ({ events: [] })) {
  answerGroupRequests(request, { local: method => ({
    'groups.capabilities': CANONICAL_GROUP_CAPABILITIES, 'groups.state': state(), 'groups.log': log()
  } as Record<string, unknown>)[method] ?? control(method) })
}

const discardCases = locales.flatMap(language => [
  ...(['withoutReceipt', 'withReceipt'] as const).flatMap(state => [
    { language, state, refusal: 'stopPending' as const, message: 'discardStopPending' as const },
    { language, state, refusal: 'unreachable' as const, message: 'discardPeerUnreachable' as const }
  ]),
  { language, state: 'offline' as const, refusal: 'stopPending' as const, message: 'discardStopPending' as const },
  { language, state: 'beforeClose' as const, refusal: 'tooOld404' as const, message: 'discardPeerUpdateRequired' as const }
])

it.each(discardCases)('offers Discard $state and keeps the deferred turn on $refusal ($language)', async ({ language, state, refusal, message }) => {
  locale.value = language
  const labels = CANONICAL_GROUP_LOCALES[language]
  let snapshot = frame(state).result
  let release!: (value: unknown) => void
  gateway(() => snapshot, method => {
    if (method === 'groups.discard') {return new Promise(resolve => { release = resolve }).then(value => {
      const response = JSON.parse(value as string)

      if (response.error) {throw jsonRpcErrorFromFrame(response.error)}
      snapshot = frame(state === 'beforeClose' ? 'afterClose' : 'afterDiscard').result

      return response.result
    })}

    return undefined
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const discard = await screen.findByRole('button', { name: 'Discard' })
  expect(screen.getByText(labels.pendingDiscard.replace('{name}', 'reviewer'))).toBeTruthy()
  fireEvent.click(discard)
  const confirm = within(screen.getByRole('alertdialog', { name: labels.discardUnknown }))
  expect(confirm.getByText(labels.discardWarning)).toBeTruthy()
  fireEvent.click(confirm.getByRole('button', { name: labels.confirmDiscard }))
  await waitFor(() => expect((discard as HTMLButtonElement).disabled).toBe(true))
  await act(async () => release(wire[refusal]))
  const alert = await screen.findByRole('alert')
  expect(alert).toHaveProperty('textContent', labels[message].replace('{name}', 'reviewer'))

  // A retry of Discard keeps the exact turn; only a confirmed result and refreshed state remove it.
  fireEvent.click(discard)
  fireEvent.click(screen.getByRole('button', { name: labels.confirmDiscard }))
  await act(async () => release(state === 'beforeClose' ? wire.closed : wire.discarded))
  await waitFor(() => expect(screen.queryByRole('button', { name: 'Discard' })).toBeNull())
})

const disbandCases = locales.flatMap(language => DISBAND_FAILURES.map(failure => ({ language, ...failure })))

it.each(locales)('explains how to restore a missing member home before Disband (%s)', async language => {
  for (const code of [4001, 5114]) {
    gateway(() => ({}), method => method === 'groups.disband' ? jsonRpcErrorFromFrame({ code,
      message: 'member_home_unknown', data: { reason: 'member_home_unknown' } }) : undefined)
    await expect(disbandCanonicalGroup(binding, CANONICAL_GROUP_LOCALES[language])).rejects.toThrow('hosted_rooms.profiles')
  }
})

it.each(disbandCases)('keeps the group chat and explains Disband $name ($language)', async ({ language, name, error, detail }) => {
  locale.value = language
  const labels = CANONICAL_GROUP_LOCALES[language]
  const snapshot = frame('offline').result
  const room = snapshot.room
  const onBack = vi.fn()
  const key = registerCanonicalGroup(binding, room)
  updateGroupChat(room.name, record => ({ ...record, roomId: binding.roomId, engine: { kind: 'gateway', ...binding } }), { sync: false })
  const original = structuredClone($groupChats.get()[room.name])
  gateway(() => snapshot, method => ({ 'groups.disband': error, 'groups.retry': { retried: true } } as Record<string, unknown>)[method])
  const view = render(<CanonicalGroupWorkspace binding={binding} onBack={onBack} />)
  const disband = translateBotsIn(language)('group.disbandAction') as string
  fireEvent.click(await screen.findByRole('button', { name: disband }))
  const dialog = within(await screen.findByRole('dialog'))
  fireEvent.click(dialog.getByRole('button', { name: disband }))
  expect(await dialog.findByText(labels.disbandKept.replace('{reason}', labels[detail]))).toBeTruthy()

  if (language === 'en') {
    expect(dialog.queryByText(/Its work was stopped\./) !== null).toBe(['peer unreachable', 'revoke unsupported'].includes(name))
  }

  expect((dialog.getByRole('button', { name: disband }) as HTMLButtonElement).disabled).toBe(false)
  expect($canonicalGroupBindings.get()[key]).toMatchObject(binding)
  expect($groupChats.get()[room.name]).toEqual(original)
  expect(onBack).not.toHaveBeenCalled()

  if (name === 'stop pending') {
    fireEvent.click(dialog.getByRole('button', { name: 'Cancel' }))
    expect(view.baseElement.textContent).not.toContain(labels.cycleStopped)
    expect(screen.getByRole('button', { name: 'Discard' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    const { member_id, task_id, execution_generation } = snapshot.driver_status.pending_actions[0]
    await waitFor(() => expect(request).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'local', profile: 'default' }), 'groups.retry', {
      room_id: binding.roomId, profile: 'default', member_id, task_id, execution_generation
    }))
  }
})

const stopAdvice = {
  en: ['awaiting confirmation', 'Stop requested'], ja: ['確認待ち', '停止を要求しました'],
  zh: ['等待确认', '已请求停止'], 'zh-hant': ['等待確認', '已要求停止'],
  ar: ['تنتظر التأكيد', 'طُلب الإيقاف'], ru: ['ожидающие подтверждения', 'Остановка запрошена'],
  fr: ['en attente de confirmation', 'Arrêt demandé'], de: ['Bestätigung aussteht', 'Stopp angefordert'],
  es: ['pendientes de confirmación', 'Parada solicitada']
}

it.each(locales.flatMap(language => (['stopZero', 'stopMixed'] as const).map(response => ({ language, response }))))(
  'keeps unconfirmed turns and recovery actions after $response ($language)', async ({ language, response }) => {
    locale.value = language
    const labels = CANONICAL_GROUP_LOCALES[language]
    let snapshot = frame('beforeStopMixed').result
    let stopped = false
    gateway(() => snapshot, method => {
      if (method === 'groups.stop') {
        stopped = true
        snapshot = frame(response === 'stopZero' ? 'afterStop' : 'afterStopMixed').result

        return frame(response).result
      }

      return method === 'groups.retry' ? { retried: true } : undefined
    }, () => stopped ? frame('stopLog').result : { events: [] })
    const view = render(<CanonicalGroupWorkspace binding={binding} />)
    fireEvent.click(await screen.findByRole('button', { name: labels.stopGroup }))
    expect((await screen.findByText(labels.stopped)).textContent).toContain(stopAdvice[language][0])
    expect(await screen.findAllByText(stopAdvice[language][1])).not.toHaveLength(0)
    expect(view.baseElement.textContent).not.toContain(labels.cycleStopped)
    expect(screen.getByRole('button', { name: 'Discard' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    const { member_id, task_id, execution_generation } = snapshot.driver_status.pending_actions[0]
    await waitFor(() => expect(request).toHaveBeenCalledWith(expect.anything(), 'groups.retry', {
      profile: 'default', room_id: 'linked', member_id, task_id, execution_generation
    }))
  }
)

it('offers only Retry for a deferred turn on another local profile', async () => {
  locale.value = 'en'
  const snapshot = frame('localDeferred').result
  gateway(() => snapshot, () => undefined)
  render(<CanonicalGroupWorkspace binding={binding} />)
  expect(await screen.findByRole('button', { name: 'Retry' })).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Discard' })).toBeNull()
  expect(screen.getByText('reviewer’s turn has not finished. Retry it?')).toBeTruthy()
})

it('shows completed Stop only after the gateway has no waiting work', async () => {
  locale.value = 'en'
  gateway(() => frame('afterDiscard').result, () => undefined, () => frame('stopLog').result)
  render(<CanonicalGroupWorkspace binding={binding} />)
  expect(await screen.findByText(CANONICAL_GROUP_LOCALES.en.cycleStopped, { exact: true })).toBeTruthy()
})
