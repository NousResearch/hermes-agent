/**
 * The Skills Hub picker embeds the real hub page in an iframe; the page posts
 * `{type:'hermes-skill-pick'}` back and the plugin installs via skills.manage.
 *
 * The handler used to check only `event.origin`, so ANY window on the hub
 * origin — an OAuth popup that had navigated back there, for instance — could
 * drive an install while the picker was open, and whatever string it sent
 * reached skills.manage as the install identifier. Picks are now pinned to OUR
 * frame's contentWindow and the identifier is charset-checked.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

const HUB_ORIGIN = 'https://hermes-agent.nousresearch.com'

const mocks = vi.hoisted(() => ({
  notify: vi.fn(),
  notifyError: vi.fn(),
  request: vi.fn(async (_method: string, _params: Record<string, unknown>) => ({})),
  requestProfile: vi.fn(async (_route: unknown, _method: string, _params: Record<string, unknown>) => ({}))
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()

  return {
    ...original,
    usePluginI18n: () => translateBots,
    host: {
      ...original.host,
      notify: mocks.notify,
      notifyError: mocks.notifyError,
      request: mocks.request,
      requestProfile: mocks.requestProfile
    }
  }
})

const { HubSkillsSection } = await import('./skills-hub')

interface PickMessage {
  identifier?: string
  name?: string
  type?: string
}

/** Mount the section with the hub browser open and hand back its frame. */
function openHubBrowser() {
  const { container } = render(<HubSkillsSection />)

  fireEvent.click(screen.getByRole('button', { name: /browse the full hub/i }))

  const frame = container.querySelector('iframe')

  expect(frame).toBeTruthy()

  return frame as HTMLIFrameElement
}

function postPick(data: PickMessage, options: { origin?: string; source?: null | Window }) {
  window.dispatchEvent(
    new MessageEvent('message', {
      data,
      origin: options.origin ?? HUB_ORIGIN,
      source: options.source ?? null
    })
  )
}

function installCalls() {
  return mocks.request.mock.calls.filter(([, params]) => params.action === 'install')
}

function routedInstallCalls() {
  return mocks.requestProfile.mock.calls.filter(([, , params]) => params.action === 'install')
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  cleanup()
})

describe('hub pick messages', () => {
  it('installs the picked skill when it comes from our own frame', () => {
    const frame = openHubBrowser()

    postPick(
      { identifier: 'nous/web-research', name: 'Web Research', type: 'hermes-skill-pick' },
      {
        source: frame.contentWindow
      }
    )

    expect(installCalls()).toEqual([['skills.manage', { action: 'install', query: 'nous/web-research' }]])
  })

  it('routes an existing source-scoped bot install through its owner connection', async () => {
    const { container } = render(
      <HubSkillsSection
        bot={{
          name: 'worker',
          remoteSource: true,
          route: {
            connectionId: 'remote-a',
            mode: 'remote',
            profile: 'worker',
            targetProfile: 'backend-worker'
          },
          sourceScoped: true
        }}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: /browse the full hub/i }))
    const frame = container.querySelector('iframe') as HTMLIFrameElement

    postPick(
      { identifier: 'nous/web-research', name: 'Web Research', type: 'hermes-skill-pick' },
      { source: frame.contentWindow }
    )

    await waitFor(() => expect(routedInstallCalls()).toHaveLength(1))
    expect(routedInstallCalls()).toEqual([
      [
        expect.objectContaining({
          connectionId: 'remote-a',
          mode: 'remote',
          profile: 'worker',
          targetProfile: 'backend-worker'
        }),
        'skills.manage',
        {
          action: 'install',
          profile: 'backend-worker',
          query: 'nous/web-research'
        }
      ]
    ])
    expect(installCalls()).toEqual([])
  })

  it('regression: ignores a same-origin window that is NOT the picker frame', () => {
    openHubBrowser()

    // Same origin, different window — the OAuth-popup shape of the hole.
    postPick(
      { identifier: 'nous/web-research', name: 'Web Research', type: 'hermes-skill-pick' },
      {
        source: window
      }
    )

    expect(installCalls()).toEqual([])
  })

  it('ignores anything posted from another origin', () => {
    const frame = openHubBrowser()

    postPick(
      { identifier: 'nous/web-research', name: 'Web Research', type: 'hermes-skill-pick' },
      {
        origin: 'https://evil.example',
        source: frame.contentWindow
      }
    )

    expect(installCalls()).toEqual([])
  })

  it('regression: refuses identifiers outside the slug charset', () => {
    const frame = openHubBrowser()

    for (const identifier of ['../../etc/passwd', 'skill; rm -rf /', '-flag', 'name with spaces', '']) {
      postPick({ identifier, name: 'Web Research', type: 'hermes-skill-pick' }, { source: frame.contentWindow })
    }

    // The empty identifier falls back to `name`, which is also off-charset.
    expect(installCalls()).toEqual([])
  })

  it('ignores messages that are not a skill pick', () => {
    const frame = openHubBrowser()

    postPick({ identifier: 'nous/web-research', type: 'oauth-callback' }, { source: frame.contentWindow })
    postPick({ identifier: 'nous/web-research', type: 'hermes-skill-pick' }, { source: frame.contentWindow })

    // The second has no `name`, so it is not a complete pick either.
    expect(installCalls()).toEqual([])
  })

  it('stops listening once the hub browser is closed', () => {
    const frame = openHubBrowser()

    fireEvent.click(screen.getByRole('button', { name: /hide the hub browser/i }))
    postPick(
      { identifier: 'nous/web-research', name: 'Web Research', type: 'hermes-skill-pick' },
      {
        source: frame.contentWindow
      }
    )

    expect(installCalls()).toEqual([])
  })
})
