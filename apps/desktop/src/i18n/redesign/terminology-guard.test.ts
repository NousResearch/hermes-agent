import { renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { useAgentSettingsCopy } from '@/app/agent-settings/copy'
import { useHistoryCopy } from '@/app/agent-settings/history-copy'
import { useOverviewActionsCopy } from '@/app/agent-settings/overview-actions-copy'
import { useAgentSettingsEditorCopy } from '@/app/agent-settings/settings-copy'
import { useAgentSettingsLayoutCopy } from '@/app/agent-settings/settings-layout-copy'
import { useCapabilitiesPageDescription } from '@/app/capabilities/page-copy'
import { agentCapabilityCopy } from '@/app/capabilities/ui/agent-capability-copy'
import { useHiddenAgentCopy } from '@/app/chat/sidebar/hidden-agent-copy'
import { useNewSessionCopy } from '@/app/chat/sidebar/simplified-new-session-copy'
import { usePageFrameCopy } from '@/app/overlays/page-frame-copy'
import { routinesCopy } from '@/app/routines/routines-copy'
import { appearanceAdvancedCopy } from '@/app/settings/appearance-advanced-copy'
import { appearanceOverviewCopy } from '@/app/settings/appearance-overview-copy'
import { redesignNavigationCopy } from '@/app/settings/redesign-navigation-copy'
import { settingsPagesCopy } from '@/app/settings/settings-pages-copy'

import { TRANSLATIONS } from '../catalog'
import { resolveTranslations } from '../registry'

import { enRedesign } from './en'

/** Every user-visible leaf as `dotted.key → text`; functions render with placeholder arguments. */
function leaves(tree: unknown, prefix = '', out: Record<string, string> = {}): Record<string, string> {
  if (typeof tree === 'string') {
    out[prefix] = tree
  } else if (typeof tree === 'function') {
    out[prefix] = String((tree as (...args: unknown[]) => unknown)('‹a›', '‹b›', '‹c›'))
  } else if (Array.isArray(tree)) {
    tree.forEach((item, index) => leaves(item, `${prefix}.${index}`, out))
  } else if (tree && typeof tree === 'object') {
    for (const [key, value] of Object.entries(tree)) {
      leaves(value, prefix ? `${prefix}.${key}` : key, out)
    }
  }

  return out
}

// The classic nouns the redesign renames: Sessions → Chats, Artifacts → Library, Profiles → Agents,
// Cron / scheduled jobs → Routines, Gateways → Machines.
const CLASSIC_TERMS = /\b(sessions?|artifacts?|cron|gateways?|profiles?|scheduled jobs?)\b/i

const keep = (reason: string, ...keys: string[]) => Object.fromEntries(keys.map(key => [key, reason]))

/** Redesign keys that keep a classic noun on purpose, grouped by why. */
const REDESIGN_KEEPS: Record<string, string> = {
  // A sign-in / auth "session" (and its token), not a Hermes conversation.
  ...keep(
    'sign-in session',
    'boot.failure.remoteDescription',
    'boot.failure.remoteSignInHint',
    'settings.gateway.remoteAuthHint',
    'settings.gateway.cloudSignedInDesc',
    'settings.gateway.cloudSignedOutMessage',
    'settings.gateway.authSignedInPassword',
    'settings.gateway.authSignedInOauth',
    'settings.gateway.tokenTitle',
    'settings.gateway.tokenDesc',
    'settings.gateway.pasteSessionToken',
    'settings.gateway.incompleteToken',
    'settings.gateway.incompleteTokenTest',
    'settings.gateway.signedOutMessage',
    'settings.billing.errors.sessionRevoked.title',
    'settings.billing.errors.sessionRevoked.message',
    'profiles.remoteOverride.tokenPlaceholder',
    'install.connectExistingDesc',
    'install.tokenTitle',
    'install.tokenDesc',
    'install.pasteSessionToken',
    'install.incompleteTokenTest'
  ),
  // A password manager stays unlocked for the app's run, not for one chat.
  ...keep(
    'password-manager unlock session',
    'settings.vault.sources.blurb',
    'settings.vault.sources.unlockedDesc',
    'settings.vault.sources.unlocked',
    'prompts.vaultUnlockDesc'
  ),
  // The OS desktop session, an MCP server state, and the app's current run.
  ...keep(
    'OS / app session',
    'settings.hudModifier.unsupportedSession',
    'skills.plugins.serverStates.no_interactive_session',
    'interfaceMode.sessionNote'
  ),
  // The browser profile and the memory "user profile" (USER.md) are not agents.
  ...keep(
    'browser or user profile',
    'settings.subpages.browserProfile',
    'settings.fieldLabels.browser.useRealProfile',
    'settings.fieldLabels.memory.userProfileEnabled',
    'settings.fieldLabels.memory.userCharLimit',
    'settings.fieldDescriptions.browser.useRealProfile',
    'settings.fieldDescriptions.memory.userProfileEnabled',
    'settings.memoryLayers.builtinDesc',
    'settings.toolsets.browserRealProfile.label',
    'settings.toolsets.browserRealProfile.description',
    'settings.toolsets.browserRealProfile.enabledTitle',
    'settings.toolsets.browserRealProfile.enabledMessage',
    'settings.toolsets.browserRealProfile.disabledTitle',
    'settings.toolsets.browserRealProfile.disabledMessage',
    'settings.toolsets.browserRealProfile.failedSave',
    'settings.toolsets.browserRealProfile.signinsImported',
    'settings.toolsets.browserRealProfile.signinsRemote',
    'settings.toolsets.browserRealProfile.prompt.body',
    'settings.toolsets.browserRealProfile.prompt.bulletLiveProfile',
    'settings.toolsets.browserRealProfile.prompt.enable',
    'commandCenter.maintenance.userFile',
    'commandCenter.maintenance.resetUser'
  ),
  // Literal CLI commands, an env var, a URL path, and the Team Gateway product name.
  ...keep(
    'literal command / identifier / product name',
    'promptNotices.desktopCommands.exportRemote',
    'promptNotices.desktopCommands.importRemote',
    'profiles.exportNeedsHost',
    'desktop.profileStatus',
    'settings.gateway.webappOauthPaste.placeholder',
    'messaging.relayLink.title',
    'messaging.relayLink.managed',
    'messaging.relayLink.managedNote'
  ),
  // A routine HAS a schedule; "cron" names the expression syntax its input accepts.
  ...keep(
    'cron expression syntax',
    'cron.scheduleHints.custom',
    'cron.createDesc',
    'cron.customHint',
    'tips.items.cron.text'
  )
}

/** Copy-module leaves that keep a classic noun on purpose. */
const COPY_KEEPS: Record<string, string> = {
  ...keep(
    'browser or user profile',
    'agentSettingsLayout.summaries.browser',
    'agentSettingsLayout.rows.memory.user_profile_enabled.0',
    'agentSettingsLayout.rows.memory.user_profile_enabled.1'
  ),
  ...keep('password-manager unlock session', 'settingsPages.vault.managersDesc'),
  // Developer mode documents the schedule's input syntax by its name.
  ...keep('cron expression syntax', 'routines.customHintDeveloper')
}

/** Redesign-only copy that lives outside the catalog (each module holds all nine locales). */
function redesignCopyModules(): Record<string, unknown> {
  const hook = <T>(use: () => T): T => renderHook(use).result.current

  return {
    agentSettings: hook(useAgentSettingsCopy),
    agentSettingsEditor: hook(useAgentSettingsEditorCopy),
    agentSettingsLayout: hook(useAgentSettingsLayoutCopy),
    agentCapability: agentCapabilityCopy('en'),
    appearanceAdvanced: appearanceAdvancedCopy('en'),
    appearanceOverview: appearanceOverviewCopy('en'),
    capabilitiesPage: hook(useCapabilitiesPageDescription),
    hiddenAgent: hook(useHiddenAgentCopy),
    history: hook(useHistoryCopy),
    newSession: hook(useNewSessionCopy),
    overviewActions: hook(useOverviewActionsCopy),
    pageFrame: hook(usePageFrameCopy),
    redesignNavigation: redesignNavigationCopy('en'),
    routines: routinesCopy('en'),
    settingsPages: settingsPagesCopy('en'),
    simplifiedAgents: TRANSLATIONS.en.simplifiedAgents
  }
}

/** Base keys only the redesigned shell renders: the base may hold redesign words there. */
const REDESIGN_ONLY_BASE = [
  'simplifiedAgents.',
  'firstRun.guide.',
  'settings.appearance.redesign.',
  'skills.openToolLibrary'
]

// Words that exist only in the redesign: classic must not gain them anywhere `main` did not have them.
const REDESIGN_TERMS = /\b(routines?|library)\b/i

/** Classic keys whose `main` wording already used one of those words, in another sense. */
const CLASSIC_KEEPS: Record<string, string> = {
  'settings.localModels.sideloadAlreadyPresent': "a local model already downloaded ('in your library')"
}

describe('redesign terminology', () => {
  it('leaves no classic noun in the redesigned English catalog outside the commented keeps', () => {
    const offenders = Object.entries(leaves(resolveTranslations('en', 'redesign')))
      .filter(([key, text]) => CLASSIC_TERMS.test(text) && !(key in REDESIGN_KEEPS))
      .map(([key, text]) => `${key}: ${text}`)

    expect(offenders).toEqual([])
  })

  it('keeps every allowlisted redesign key real and still carrying its classic noun', () => {
    const redesign = leaves(resolveTranslations('en', 'redesign'))
    const stale = Object.keys(REDESIGN_KEEPS).filter(key => !CLASSIC_TERMS.test(redesign[key] ?? ''))

    expect(stale).toEqual([])
  })

  it('leaves no classic noun in the redesign-only copy modules', () => {
    const offenders = Object.entries(leaves(redesignCopyModules()))
      .filter(([key, text]) => CLASSIC_TERMS.test(text) && !(key in COPY_KEEPS))
      .map(([key, text]) => `${key}: ${text}`)

    expect(offenders).toEqual([])
  })

  it('takes redesign-only words into classic from nowhere: the overlay is their only source', () => {
    const offenders = Object.entries(leaves(TRANSLATIONS.en))
      .filter(([key, text]) => REDESIGN_TERMS.test(text) && !(key in CLASSIC_KEEPS))
      .filter(([key]) => !REDESIGN_ONLY_BASE.some(prefix => key.startsWith(prefix)))
      .map(([key, text]) => `${key}: ${text}`)

    expect(offenders).toEqual([])
    expect(Object.values(leaves(enRedesign)).some(text => /\bRoutines\b/.test(text))).toBe(true)
    expect(Object.values(leaves(enRedesign)).some(text => /\bLibrary\b/.test(text))).toBe(true)
  })
})
