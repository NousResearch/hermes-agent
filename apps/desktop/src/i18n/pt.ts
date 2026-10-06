import { defineLocale } from './define-locale'
import { introPt } from './intro-pt'
import { ptAssistant } from './pt_assistant'
import { ptBoot } from './pt_boot'
import { ptCapabilities } from './pt_capabilities'
import { ptChat } from './pt_chat'
import { ptChrome } from './pt_chrome'
import { ptCommandCenter } from './pt_command_center'
import { ptCommon } from './pt_common'
import { ptModels } from './pt_models'
import { ptSettingsA } from './pt_settings_a'
import { ptSettingsB } from './pt_settings_b'

// Brazilian Portuguese desktop catalog, written as partial overrides merged over
// `en` and split by topic (`pt_<topic>.ts`) like the other locales: keys left out
// fall back to English, and every key stays type-checked against `Translations`,
// so en.ts additions can't drift silently. Shares the `pt` id with the web
// catalog and the backend's `locales/pt.yaml`.
export const pt = defineLocale({
  ...ptCommon,
  ...ptBoot,
  ...ptModels,
  ...ptCapabilities,
  ...ptCommandCenter,
  ...ptChat,
  ...ptChrome,
  ...ptAssistant,
  intro: introPt,
  settings: { ...ptSettingsA, ...ptSettingsB }
})
