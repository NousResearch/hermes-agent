/**
 * The New Bot dialog validates its slug with the SDK's PROFILE_ID_ARG_RE, which re-exports the one
 * shared owner (state-24, #155595). The slugifier only emits `[a-z0-9-]`, so the reachable proof is
 * owner-driven rejection: under a swapped owner a slug the old private copy accepted is refused.
 */

import { fireEvent, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { createAgentHarness } from './create-agent-test-utils'
import { translateBots } from './i18n-test-helper'

const mocks = await vi.hoisted(async () => (await import('./create-agent-test-utils')).createMocks())

vi.mock('@hermes/shared/profile-id', () => ({ PROFILE_ID_ARG_RE: /^zz-[a-z]+$/ }))
vi.mock('@hermes/plugin-sdk', async importOriginal =>
  (await import('./create-agent-test-utils')).sdkMock(await importOriginal(), mocks, () => translateBots)
)
vi.mock('./canonical-chat', () => ({ createCanonicalChat: mocks.createCanonicalChat }))
vi.mock('./profile-ops', () => ({ deleteBot: mocks.deleteBot }))
vi.mock('./data', async importOriginal => ({ ...(await importOriginal<object>()), saveBotMeta: mocks.saveBotMeta }))

const { renderDialog } = createAgentHarness(mocks)

describe('the bot name check follows the shared profile-id owner', () => {
  it.each([
    ['inbox-triage', true],
    ['zz-custom', false]
  ])('%s → Create disabled: %s', async (name, disabled) => {
    await renderDialog(true)
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: name } })

    expect((screen.getByRole('button', { name: 'Create bot' }) as HTMLButtonElement).disabled).toBe(disabled)
  })
})
