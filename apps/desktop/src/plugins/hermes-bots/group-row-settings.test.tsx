import type * as HermesSdk from '@hermes/plugin-sdk'
import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'

import { GroupRow } from './bot-row'
import { translateBots } from './i18n-test-helper'

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, usePluginI18n: () => translateBots }
})

it('opens the existing Group settings from the roster context menu', async () => {
  const onSettings = vi.fn()
  const props = {
    active: false, group: 'Planning', members: [], needsYou: false,
    onDisband: vi.fn(), onOpen: vi.fn(), onSettings
  }
  render(<GroupRow {...props} />)
  fireEvent.contextMenu(screen.getByRole('button'))
  fireEvent.click(await screen.findByText('Group settings'))
  expect(onSettings).toHaveBeenCalledWith({ members: [], name: 'Planning' })
})
