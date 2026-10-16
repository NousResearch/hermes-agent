import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  buildTrayMenuTemplate,
  DEFAULT_TRAY_MENU_LABELS,
  sanitizeTrayPreferences,
  TRAY_MENU_ACTIONS,
  type TrayMenuEntry
} from './tray-access'

const actionOf = (entry: TrayMenuEntry) => entry.action
const labelOf = (entry: TrayMenuEntry) => ('type' in entry && entry.type === 'separator' ? undefined : entry.label)

test('tray preferences default to a fresh-conversation click and no login item', () => {
  assert.deepEqual(sanitizeTrayPreferences(undefined), {
    launchAtLogin: false,
    openNewConversationOnClick: true
  })
})

test('tray preferences never throw on garbage, and fall back to the shipped defaults', () => {
  for (const raw of [null, 'yes', 42, [], {}]) {
    assert.deepEqual(sanitizeTrayPreferences(raw), {
      launchAtLogin: false,
      openNewConversationOnClick: true
    })
  }

  // An absent value falls back to the default, but a present WRONG type does
  // not read as "on": a truthy 1 or "yes" is not the user's `true`.
  assert.deepEqual(sanitizeTrayPreferences({ launchAtLogin: 'yes', openNewConversationOnClick: 1 }), {
    launchAtLogin: false,
    openNewConversationOnClick: false
  })

  assert.deepEqual(sanitizeTrayPreferences({ launchAtLogin: true, openNewConversationOnClick: false }), {
    launchAtLogin: true,
    openNewConversationOnClick: false
  })
})

test('the tray menu is New conversation / Open Hermes / Settings / Quit, in that order', () => {
  const entries = buildTrayMenuTemplate()

  assert.deepEqual(entries.map(actionOf), ['new-conversation', 'open-app', 'settings', undefined, 'quit'])

  // Every action the menu can reach is one the router knows.
  for (const action of entries.map(actionOf)) {
    if (action) {
      assert.ok(TRAY_MENU_ACTIONS.includes(action))
    }
  }

  // Quit is the only item behind a separator, so it can never be hit by
  // muscle memory meant for the row above it.
  assert.equal(actionOf(entries.at(-1)!), 'quit')
  assert.equal(actionOf(entries.at(-2)!), undefined)
})

test('the tray menu labels are caller-overridable, and default to English', () => {
  assert.deepEqual(entriesLabels(buildTrayMenuTemplate()), [
    DEFAULT_TRAY_MENU_LABELS.newConversation,
    DEFAULT_TRAY_MENU_LABELS.openApp,
    DEFAULT_TRAY_MENU_LABELS.settings,
    undefined,
    DEFAULT_TRAY_MENU_LABELS.quit
  ])

  const localized = buildTrayMenuTemplate({
    newConversation: 'Nouvelle conversation',
    openApp: 'Ouvrir Hermes',
    quit: 'Quitter Hermes',
    settings: 'Paramètres'
  })

  assert.equal(labelOf(localized[0]), 'Nouvelle conversation')
  assert.equal(labelOf(localized[4]), 'Quitter Hermes')
  // The separator carries no label, ever.
  assert.equal(labelOf(localized[3]), undefined)
})

function entriesLabels(entries: TrayMenuEntry[]) {
  return entries.map(labelOf)
}
