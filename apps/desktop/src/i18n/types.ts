// Desktop i18n type contract.
//
// `Translations` is the single source of truth for every translatable string
// surface. Fully translated locale files may satisfy this interface directly;
// partial locales should use `defineLocale()` so missing desktop-only strings
// fall back to English while new keys remain type-checked.
//
// The catalog below is sharded by topic: each `types_<topic>.ts` sibling owns
// one domain's members. Find a string surface there, not here.

import type { ArtifactsTranslations } from './types_artifacts'
import type { AssistantTranslations } from './types_assistant'
import type { BootTranslations } from './types_boot'
import type { CapabilitiesTranslations } from './types_capabilities'
import type { ChatTranslations } from './types_chat'
import type { ChromeTranslations } from './types_chrome'
import type { CommandCenterTranslations } from './types_command_center'
import type { CommonTranslations } from './types_common'
import type { ConnectorsTranslations } from './types_connectors'
import type { DiagnosticsTranslations } from './types_diagnostics'
import type { SettingsTranslations } from './types_settings'

export type Locale = 'en' | 'zh' | 'zh-hant' | 'ja' | 'ar' | 'ru'

export interface Translations
  extends
    ArtifactsTranslations,
    AssistantTranslations,
    BootTranslations,
    CapabilitiesTranslations,
    ChatTranslations,
    ChromeTranslations,
    CommandCenterTranslations,
    CommonTranslations,
    ConnectorsTranslations,
    DiagnosticsTranslations,
    SettingsTranslations {}
