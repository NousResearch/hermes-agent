import { afterEach, expect, it } from 'vitest'

import {
  $daybreakModelChoices,
  adoptDraftDaybreakSelection,
  daybreakKeyFor,
  daybreakModelChoiceFor,
  daybreakSelectionFor,
  setDaybreakModelChoice,
  setDaybreakSelection
} from './daybreak'
import { $activeGatewayProfile, $newChatProfile, $newChatRoute } from './profile'
import { $connection, $currentModel, $currentProvider } from './session'

afterEach(() => {
  $daybreakModelChoices.set({})
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  $activeGatewayProfile.set('default')
  $connection.set(null)
})

it('does not carry an unsent Daybreak draft into another profile', () => {
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  $connection.set(null)
  $activeGatewayProfile.set('security')
  setDaybreakSelection(null, true)

  $activeGatewayProfile.set('general')
  expect(daybreakSelectionFor(null)).toBeUndefined()

  $activeGatewayProfile.set('security')
  expect(daybreakSelectionFor(null)).toBe(true)
})

it('keeps an unselected model row choice in its conversation until the draft is sent', () => {
  const draftKey = daybreakKeyFor(null)
  setDaybreakModelChoice(null, 'openai-codex::gpt-6-luna', true)

  expect(daybreakSelectionFor(null)).toBeUndefined()
  expect(daybreakModelChoiceFor('other-session', 'openai-codex::gpt-6-luna')).toBeUndefined()

  adoptDraftDaybreakSelection('stored-1', draftKey)
  expect(daybreakModelChoiceFor(null, 'openai-codex::gpt-6-luna')).toBeUndefined()
  expect(daybreakModelChoiceFor('stored-1', 'openai-codex::gpt-6-luna')).toBe(true)
})

it('keeps model choices independent across switches and restores the original model choice', () => {
  $currentProvider.set('openai-codex')
  $currentModel.set('gpt-6-sol')
  setDaybreakSelection(null, true)
  $currentModel.set('gpt-6-astra')
  expect(daybreakSelectionFor(null)).toBeUndefined()
  setDaybreakSelection(null, false)
  $currentModel.set('gpt-6-sol')
  expect(daybreakSelectionFor(null)).toBe(true)
  $currentModel.set('gpt-6-astra')
  expect(daybreakSelectionFor(null)).toBe(false)
})
