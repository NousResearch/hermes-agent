import { defineLocale, type TranslationOverrides } from './define-locale'
import { introPtBr } from './intro-pt-br'
import { ptBrAppTour } from './pt-br_app_tour'
import { ptBrBilling } from './pt-br_billing'
import { ptBrBoot } from './pt-br_boot'
import { ptBrCatalog } from './pt-br_catalog'
import { ptBrChatOverrides } from './pt-br_chat'
import { ptBrChromeOverrides } from './pt-br_chrome'
import { ptBrConnectorCard } from './pt-br_connector_card'
import { ptBrGatewayGroupOpen, ptBrSkillDeepLink } from './pt-br_deep_links'
import { ptBrExternalOpenFailed } from './pt-br_external_open'
import { ptBrFirstRun } from './pt-br_first_run'
import { ptBrNotices } from './pt-br_notices'
import { ptBrOnboarding } from './pt-br_onboarding'
import { ptBrSettingsOverrides } from './pt-br_settings'
import { ptBrSharedMetrics } from './pt-br_shared_metrics'
import { ptBrShellOverrides } from './pt-br_shell'
import { ptBrWorkspaceOverrides } from './pt-br_workspace'

export const ptBrOverrides = {
  ...ptBrChromeOverrides,
  ...ptBrSettingsOverrides,
  ...ptBrWorkspaceOverrides,
  ...ptBrChatOverrides,
  ...ptBrShellOverrides,
  ...ptBrNotices,
  ...ptBrBilling,
  externalOpenFailed: ptBrExternalOpenFailed,
  sharedMetrics: ptBrSharedMetrics,
  appTour: ptBrAppTour,
  firstRun: ptBrFirstRun,
  intro: introPtBr,
  connectors: ptBrConnectorCard,
  catalog: ptBrCatalog,
  boot: ptBrBoot.boot,
  skillDeepLink: ptBrSkillDeepLink,
  gatewayGroupOpen: ptBrGatewayGroupOpen,
  onboarding: ptBrOnboarding
} satisfies TranslationOverrides

export const ptBr = defineLocale(ptBrOverrides)
