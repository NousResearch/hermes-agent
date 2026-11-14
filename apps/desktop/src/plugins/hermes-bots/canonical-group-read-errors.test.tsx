import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { answerGroupRequests, CANONICAL_GROUP_CAPABILITIES, localGatewaySdkMock } from './canonical-group-contract-test-utils'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { translateBotsIn } from './i18n-test-helper'

const { request, locale } = vi.hoisted(() => ({ request: vi.fn(), locale: { value: 'en' } }))
vi.mock('@hermes/plugin-sdk', async importOriginal =>
  localGatewaySdkMock(await importOriginal<typeof HermesSdk>(), request, () => translateBotsIn(locale.value)))
afterEach(() => { cleanup(); request.mockReset(); localStorage.clear() })

it.each(Object.keys(CANONICAL_GROUP_LOCALES) as (keyof typeof CANONICAL_GROUP_LOCALES)[])('retries both state and history on Refresh (%s)', async language => {
  locale.value = language
  let failing = 'groups.state'
  let timedOut = true
  const room = { room_id: 'read-error', name: 'Read recovery', members: [] }
  answerGroupRequests(request, { local: method => {
    if (method === failing) {throw new Error(timedOut ? `request timed out after 30s: ${method}` : `socket closed: ${method}`)}

    return ({ 'groups.capabilities': CANONICAL_GROUP_CAPABILITIES, 'groups.state': { room, driver_status: { running: true } },
      'groups.log': { events: [] } } as Record<string, unknown>)[method]
  } })
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'default', roomId: room.room_id }} />)
  const alert = await screen.findByRole('alert')
  expect(alert.textContent).not.toMatch(/request timed out|groups\.(state|log)/)
  failing = 'groups.log'
  await act(async () => { fireEvent.click(within(alert).getByRole('button')) })
  expect(screen.getByRole('alert').textContent).not.toMatch(/request timed out|groups\.(state|log)/)
  timedOut = false
  await act(async () => { fireEvent.click(within(screen.getByRole('alert')).getByRole('button')) })
  expect(screen.getByRole('alert').textContent).toContain(CANONICAL_GROUP_LOCALES[language].groupsUnreachable)
  expect(screen.getByRole('alert').textContent).not.toContain('socket closed')
  failing = ''
  await act(async () => { fireEvent.click(within(screen.getByRole('alert')).getByRole('button')) })
  expect(screen.queryByRole('alert')).toBeNull()
  expect(screen.getByText(room.name)).toBeTruthy()
})

it('says a Retry was refused because the bot may still be working, not its reason code', async () => {
  locale.value = 'en'
  const room = { room_id: 'busy', name: 'Busy', members: [{ member_id: 'a', profile: 'a', handle: 'a', display_name: 'Ada' }] }
  const action = { kind: 'retry', task_id: 't', member_id: 'a', execution_generation: 1 }

  answerGroupRequests(request, { local: method => {
    if (method === 'groups.retry') {
      throw Object.assign(new Error('session_busy'), { code: 4001, data: {
        reason: 'session_busy', field: 'home', detail: 'an older home of this profile may still run this turn: /a/b' } })
    }

    return ({ 'groups.capabilities': CANONICAL_GROUP_CAPABILITIES, 'groups.log': { events: [], has_more: false },
      'groups.state': { room, driver_status: { running: true, working: false, pending_actions: [action] } } } as Record<string, unknown>)[method]
  } })
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'default', roomId: room.room_id }} />)
  const retry = await screen.findByRole('button', { name: 'Retry' })
  await act(async () => { fireEvent.click(retry) })
  await screen.findByText(CANONICAL_GROUP_LOCALES.en.retryBusy.replace('{name}', 'Ada'))
  expect(window.document.body.textContent).not.toContain('session_busy')
})
