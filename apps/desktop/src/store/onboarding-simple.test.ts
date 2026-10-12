import { afterEach, describe, expect, it, vi } from 'vitest'

import { setInterfaceMode } from '@/store/interface-mode'
import * as notifications from '@/store/notifications'

import {
  $desktopOnboarding,
  type DesktopOnboardingState,
  type OnboardingContext,
  saveOnboardingApiKey
} from './onboarding'

function firstRun(): DesktopOnboardingState {
  return {
    configured: false,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: null,
    reason: null,
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false,
    freeTierReady: false
  }
}

function installApi(): string[] {
  const paths: string[] = []

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      api: async ({ path }: { path: string }) => {
        paths.push(path)

        if (path.startsWith('/api/model/options')) {
          return { providers: [{ slug: 'fixture', name: 'Fixture', models: ['fixture-model'] }] }
        }

        if (path.startsWith('/api/model/recommended-default')) {
          return { model: 'fixture-model' }
        }

        return { ok: true }
      }
    }
  })

  return paths
}

function readyContext(onCompleted: () => void): OnboardingContext {
  return {
    onCompleted,
    requestGateway: async method => (method === 'setup.status' ? { provider_configured: true } : { ok: true }) as never
  }
}

afterEach(() => {
  setInterfaceMode('advanced')
  window.localStorage.clear()
  $desktopOnboarding.set(firstRun())
  vi.restoreAllMocks()
})

describe('completeWithModelConfirm by interface mode', () => {
  it('lands Simple straight in chat with the default model already saved, without a toast', async () => {
    setInterfaceMode('simple')
    $desktopOnboarding.set(firstRun())
    const paths = installApi()
    const onCompleted = vi.fn()
    const notify = vi.spyOn(notifications, 'notify')

    await expect(
      saveOnboardingApiKey('FIXTURE_API_KEY', 'fake-key', 'Fixture', readyContext(onCompleted))
    ).resolves.toMatchObject({ ok: true })

    expect(paths).toContain('/api/model/set')
    expect($desktopOnboarding.get()).toMatchObject({ configured: true, flow: { status: 'idle' } })
    expect(onCompleted).toHaveBeenCalledOnce()
    // Landing in chat is the confirmation; a toast would name the backend catalog entry.
    expect(notify).not.toHaveBeenCalledWith(expect.objectContaining({ title: 'Hermes is ready' }))
  })

  it('still asks Advanced to confirm the model', async () => {
    $desktopOnboarding.set(firstRun())
    installApi()
    const onCompleted = vi.fn()

    await saveOnboardingApiKey('FIXTURE_API_KEY', 'fake-key', 'Fixture', readyContext(onCompleted))

    expect($desktopOnboarding.get().flow).toMatchObject({ status: 'confirming_model', currentModel: 'fixture-model' })
    expect(onCompleted).not.toHaveBeenCalled()
  })
})
