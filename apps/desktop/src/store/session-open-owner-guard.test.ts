import { afterEach, expect, it } from 'vitest'

import { $notifications, clearNotifications } from './notifications'
import { _resetSessionOwnerHintsForTests, setSelectedStoredSessionId, setSessionOwnerHint } from './session'
import { canOpenSessionOwner } from './session-states'
import { $shellChoice } from './shell-variant'

const id = 'shared-session'

afterEach(() => {
  setSelectedStoredSessionId(null)
  _resetSessionOwnerHintsForTests({ storage: true })
  clearNotifications()
  $shellChoice.set(null)
})

function openSameIdFromAnotherGateway(): boolean {
  setSelectedStoredSessionId(id)
  setSessionOwnerHint(id, { connectionId: 'local', profile: 'default' })

  return canOpenSessionOwner(id, { connectionId: 'remote', profile: 'default' })
}

// The redesign lists every profile's chats in one sidebar, so a same-id row of another gateway must
// not retarget the open chat. Classic lists one profile and opens as main does: no refusal toast.
it('opens as main does in classic', () => {
  $shellChoice.set(false)

  expect(openSameIdFromAnotherGateway()).toBe(true)
  expect($notifications.get()).toEqual([])
})

it('refuses to retarget an open chat to another owner in the redesign', () => {
  $shellChoice.set(true)

  expect(openSameIdFromAnotherGateway()).toBe(false)
  expect($notifications.get().map(notification => notification.title)).toEqual(['Could not open chat'])
})
