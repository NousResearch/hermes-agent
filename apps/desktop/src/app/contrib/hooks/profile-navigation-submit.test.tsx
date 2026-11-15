import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import {
  $profileNavigationRequest,
  cancelProfileNavigationRestore,
  requestProfileNavigationRestore
} from '@/store/profile-navigation'
import type * as QuickEntry from '@/store/quick-entry'

import { actRender, Harness, type HarnessHandle } from '../../session/hooks/use-prompt-actions/index-test-utils'

import { useQuickEntryBridge } from './use-quick-entry-bridge'

const bridge = vi.hoisted(() => ({ submit: null as Parameters<typeof QuickEntry.setQuickEntrySubmitHandler>[0] }))
vi.mock('@/store/quick-entry', async importOriginal => ({
  ...(await importOriginal<typeof QuickEntry>()),
  setQuickEntrySubmitHandler: (submit: typeof bridge.submit) => {
    bridge.submit = submit
  }
}))

afterEach(() => {
  cancelProfileNavigationRestore()
  cleanup()
})

it('quick-entry current-chat submit cancels restore before the prompt pipeline awaits', async () => {
  let handle!: HarnessHandle
  const create = vi.fn(() => new Promise<null | string>(() => {}))
  await actRender(
    <Harness
      activeSessionId={null}
      createBackendSessionForSend={create}
      onReady={value => {
        handle = value
      }}
      refreshSessions={async () => {}}
      requestGateway={async () => ({}) as never}
      storedSessionId={null}
    />
  )
  renderHook(() => useQuickEntryBridge({ submitText: handle.submitTextRaw, submitTextToNewSession: vi.fn() }))
  act(() => {
    requestProfileNavigationRestore('coder')
    void bridge.submit!({ correlationId: 'restore-test', target: 'current', text: 'from HUD' })
  })
  expect($profileNavigationRequest.get()).toBeNull()
})
