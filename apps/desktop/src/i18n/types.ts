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

// Moved to their topic shards; re-exported so existing `@/i18n/types`
// importers keep working.
export type { ErrorCardCopy, ToolTitleKey } from './types_assistant'

export type BundledLocale = 'en' | 'zh' | 'zh-hant' | 'ja' | 'ar' | 'ru' | 'fr' | 'de' | 'es'

/** Any language id the app can render: a bundled locale, or one a plugin /
 *  the backend registered at runtime (`registerAppLocale`). Lowercase
 *  BCP-7-ish (`pl`, `pt-br`). Resolve strings through the registry, never
 *  by indexing `TRANSLATIONS` directly. */
export type Locale = string

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