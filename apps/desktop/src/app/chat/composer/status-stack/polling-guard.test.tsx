import { cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { resetBackgroundPollingGuard } from '@/store/composer-status'
import { $gateway } from '@/store/gateway'

import { ComposerStatusStack } from './index'

// The stack measures itself into a surface var — jsdom has no ResizeObserver.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', ResizeObserverStub)

const SID = 'sess-dead-runtime'

function renderStack() {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerStatusStack queue={null} sessionId={SID} />
      </I18nProvider>
    </MemoryRouter>
  )
}

// #98434: a boot-restored tile can stay bound to a dead runtime id and remount
// repeatedly (no genuine rebind ever happens). The mount effect used to clear
// the gone-polling latch on every mount, so each remount re-armed the 4001
// storm against that id forever.
describe('ComposerStatusStack dead-runtime remount', () => {
  beforeEach(() => {
    resetBackgroundPollingGuard()
  })

  afterEach(() => {
    cleanup()
    $gateway.set(null as never)
    resetBackgroundPollingGuard()
  })

  it('does not re-poll process.list after a remount once the session is latched gone', async () => {
    const request = vi.fn(async (method: string) => {
      if (method === 'process.list') {
        throw new Error('session not found')
      }

      return {}
    })

    const processListCalls = () => request.mock.calls.filter(([method]) => method === 'process.list').length

    $gateway.set({ request } as never)

    const first = renderStack()
    await Promise.resolve()
    await Promise.resolve()

    expect(processListCalls()).toBe(1)

    first.unmount()

    const second = renderStack()
    await Promise.resolve()
    await Promise.resolve()

    // Before the fix: the mount effect cleared the latch, so this remount
    // re-fired process.list against the same dead id.
    expect(processListCalls()).toBe(1)

    second.unmount()
  })
})
