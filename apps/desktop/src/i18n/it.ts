import { defineLocale, type TranslationOverrides } from './define-locale'
import { introIt } from './intro-it'
import { itApp } from './it_app'
import { itAssistant } from './it_assistant'
import { itFirstRun } from './it_first_run'
import { itModelSections, itSettingsModels } from './it_models'
import { itNotices } from './it_notices'
import { itSettings } from './it_settings'
import { itWorkspace } from './it_workspace'

export const itOverrides = {
  ...itNotices,
  firstRun: itFirstRun,
  intro: introIt,
  settings: {
    ...itSettings.settings,
    ...itSettingsModels.settings
  },
  ...itModelSections,
  ...itApp,
  ...itWorkspace,
  ...itAssistant
} satisfies TranslationOverrides

export const it = defineLocale(itOverrides)
