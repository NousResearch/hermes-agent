import type { Translations } from '@/i18n'
import { NATIVE_NOTIFICATION_KINDS } from '@/store/native-notifications'
import { canUseQuickEntry } from '@/store/quick-entry'
import { TRANSLUCENCY_SUPPORTED } from '@/store/translucency'

import type { AppearanceSubpageId } from './appearance-subpages'
import type { SettingsView } from './types'

interface SettingCopy {
  description?: string
  label: string
}

export interface SettingDefinition {
  /** Rows that only exist on some platforms stay out of search, where a hit would scroll to nothing. */
  available?: () => boolean
  copy: (t: Translations) => SettingCopy
  keywords: readonly string[]
  /** The child page the row lives on; omitted when the view has no subpages. */
  subpage?: string
}

type AppearanceCopy = Translations['settings']['appearance']
type TitledKey = { [K in keyof AppearanceCopy]: K extends `${infer Base}Title` ? Base : never }[keyof AppearanceCopy]

// Most Appearance rows read `<key>Title` / `<key>Desc` straight from the
// appearance catalog; a Desc that is a formatter (uiScale) is left to the row.
const appearanceCopy =
  <K extends TitledKey>(key: K) =>
  (t: Translations): SettingCopy => {
    const catalog: Record<string, unknown> = t.settings.appearance
    const description = catalog[`${key}Desc`]

    return {
      label: t.settings.appearance[`${key}Title`],
      description: typeof description === 'string' ? description : undefined
    }
  }

const appearanceSetting = (subpage: AppearanceSubpageId, keywords: readonly string[], key: TitledKey) =>
  ({ subpage, keywords, copy: appearanceCopy(key) }) satisfies SettingDefinition

/**
 * The manifest of every hand-built settings row — the ones that are not a
 * config-schema field, a credential, or a plugin (those index themselves from
 * backend metadata). A key becomes the row's deep-link id
 * (`<view>.<kebab-case>`), its subpage routes that id, and its copy + keywords
 * feed the command palette — so a row cannot exist without being searchable,
 * and search cannot point at a row that is not there.
 *
 * List *items* (an archived chat, a saved connection, a credential) are not
 * settings and do not belong here; their pages own their own deep links.
 */
export const SETTINGS_MANIFEST = {
  appearance: {
    language: {
      subpage: 'general',
      keywords: ['locale'],
      copy: t => ({ label: t.language.label, description: t.language.description })
    },
    introSplash: appearanceSetting('general', ['splash', 'wordmark', 'empty chat', 'new chat'], 'introSplash'),
    resumeLastSession: appearanceSetting(
      'general',
      ['resume', 'reopen', 'launch', 'startup', 'last chat', 'session'],
      'resumeLastSession'
    ),
    tips: appearanceSetting('general', ['tips', 'hints', 'coach marks', 'onboarding', 'help'], 'tips'),
    tours: appearanceSetting('general', ['tour', 'walkthrough', 'guide', 'onboarding', 'help'], 'tours'),
    theme: appearanceSetting('theme', ['color mode', 'skin', 'light', 'dark'], 'theme'),
    uiScale: appearanceSetting('typography', ['zoom', 'size'], 'uiScale'),
    chatFont: appearanceSetting('typography', ['font', 'typeface', 'family', 'text'], 'chatFont'),
    terminalFont: appearanceSetting(
      'typography',
      ['font', 'monospace', 'nerd font', 'glyphs', 'shell'],
      'terminalFont'
    ),
    interfaceMode: {
      subpage: 'window-layout',
      keywords: ['simple', 'advanced', 'mode', 'interface', 'chrome', 'minimal', 'focus'],
      copy: t => ({ label: t.interfaceMode.title, description: t.interfaceMode.hint })
    },
    sessionDensity: appearanceSetting(
      'window-layout',
      ['sidebar', 'sessions', 'compact', 'comfortable', 'density'],
      'sessionDensity'
    ),
    tabStrip: appearanceSetting('window-layout', ['tabs', 'tab bar', 'strip'], 'tabStrip'),
    appActions: appearanceSetting(
      'window-layout',
      ['titlebar', 'settings gear', 'layout', 'HUD', 'left', 'right', 'tabs'],
      'appActions'
    ),
    minimizeToTray: {
      subpage: 'window-layout',
      keywords: ['tray', 'background', 'minimize', 'dock', 'taskbar', 'menu bar'],
      available: () => Boolean(window.hermesDesktop?.minimizeToTray),
      copy: t => ({ label: t.settings.config.minimizeToTrayTitle, description: t.settings.config.minimizeToTrayDesc })
    },
    translucency: {
      ...appearanceSetting('window-layout', ['opacity', 'transparent', 'glass', 'blur'], 'translucency'),
      available: () => TRANSLUCENCY_SUPPORTED
    },
    backdrop: appearanceSetting('window-layout', ['background', 'blur'], 'backdrop'),
    fileBrowser: appearanceSetting(
      'window-layout',
      ['file browser', 'files', 'file tree', 'explorer', 'right sidebar', 'panel', 'startup'],
      'fileBrowser'
    ),
    composerPopout: appearanceSetting(
      'window-layout',
      ['composer', 'floating', 'drag', 'popout', 'dock', 'input'],
      'composerPopout'
    ),
    userBubble: appearanceSetting('chat-display', ['opacity', 'transparent', 'message', 'bubble'], 'userBubble'),
    textDirection: appearanceSetting(
      'chat-display',
      ['rtl', 'ltr', 'right to left', 'left to right', 'bidi', 'arabic', 'hebrew', 'persian', 'align'],
      'textDirection'
    ),
    hideThreadTimeline: appearanceSetting(
      'chat-display',
      ['thread', 'conversation', 'timeline', 'bars', 'rail', 'navigation', 'hide'],
      'hideThreadTimeline'
    ),
    reactions: appearanceSetting('chat-display', ['emoji', 'tapback', 'react', 'reactions'], 'reactions'),
    vibeHearts: appearanceSetting('chat-display', ['hearts', 'vibe', 'celebrate', 'confetti', 'fun'], 'vibeHearts'),
    toolView: appearanceSetting('chat-display', ['tool display', 'technical'], 'toolView'),
    hideCodeDiffs: appearanceSetting(
      'chat-display',
      ['code', 'diff', 'patch', 'file edits', 'inline', 'added', 'removed'],
      'hideCodeDiffs'
    ),
    reasoningCollapsed: appearanceSetting(
      'chat-display',
      ['thinking', 'reasoning', 'collapse', 'expand', 'chain of thought'],
      'reasoningCollapsed'
    ),
    embeds: appearanceSetting('chat-display', ['external content', 'privacy'], 'embeds'),
    pet: {
      subpage: 'pet',
      keywords: ['pet', 'mascot', 'petdex', 'companion', 'buddy'],
      copy: t => ({ label: t.settings.appearance.pet.chooseTitle, description: t.settings.appearance.pet.chooseDesc })
    }
  },
  chat: {
    attachmentSize: {
      subpage: 'attachments',
      keywords: ['attachment', 'image', 'preview', 'upload', 'file size', 'limit', 'MB'],
      copy: t => ({ label: t.settings.config.attachmentSizeTitle, description: t.settings.config.attachmentSizeDesc })
    }
  },
  advanced: {
    keepAwake: {
      subpage: 'desktop',
      keywords: ['sleep', 'awake', 'caffeinate', 'idle', 'overnight', 'power'],
      copy: t => ({ label: t.settings.config.keepAwakeTitle, description: t.settings.config.keepAwakeDesc })
    },
    disableF12: {
      subpage: 'desktop',
      keywords: ['devtools', 'developer tools', 'f12', 'inspector', 'debug'],
      copy: t => ({ label: t.settings.config.disableF12Title, description: t.settings.config.disableF12Desc })
    },
    warmBotBackends: {
      subpage: 'desktop',
      keywords: ['pool', 'backends', 'bots', 'warm', 'concurrency', 'limit'],
      copy: t => ({ label: t.settings.poolLimits.warmBotBackendsTitle })
    },
    backendIdleTimeout: {
      subpage: 'desktop',
      keywords: ['pool', 'backends', 'idle', 'timeout', 'milliseconds'],
      copy: t => ({ label: t.settings.poolLimits.backendIdleTimeoutTitle })
    },
    quickEntry: {
      subpage: 'desktop',
      keywords: ['quick entry', 'global shortcut', 'spotlight', 'summon', 'prompt anywhere'],
      available: canUseQuickEntry,
      copy: t => ({ label: t.settings.quickEntry.enabledTitle, description: t.settings.quickEntry.enabledDesc })
    },
    quickEntryShortcut: {
      subpage: 'desktop',
      keywords: ['quick entry', 'shortcut', 'hotkey', 'keybind', 'chord'],
      available: canUseQuickEntry,
      copy: t => ({ label: t.settings.quickEntry.shortcutTitle, description: t.settings.quickEntry.shortcutDesc })
    }
  },
  keybinds: {
    hudModifier: {
      subpage: 'hud-gesture',
      keywords: ['HUD', 'summon', 'modifier', 'tap', 'Ctrl', 'Alt', 'Command', 'Option'],
      available: () => Boolean(window.hermesDesktop?.hudModifier),
      copy: t => ({ label: t.settings.hudModifier.title, description: t.settings.hudModifier.description })
    },
    screenshot: {
      subpage: 'screen-capture',
      keywords: ['screenshot', 'screen capture', 'window', 'attach', 'command keys'],
      available: () => Boolean(window.hermesDesktop?.screenshot),
      copy: t => ({ label: t.settings.screenshot.enabledTitle, description: t.settings.screenshot.enabledDesc })
    }
  },
  notifications: {
    enableAll: {
      subpage: 'alerts',
      keywords: ['notifications', 'alerts', 'native', 'banner', 'mute', 'do not disturb'],
      copy: t => ({ label: t.settings.notifications.enableAll, description: t.settings.notifications.enableAllDesc })
    },
    ...Object.fromEntries(
      NATIVE_NOTIFICATION_KINDS.map(kind => [
        `kind-${kind}`,
        {
          subpage: 'alerts',
          keywords: ['notification', 'alert', kind],
          copy: (t: Translations) => ({
            label: t.settings.notifications.kinds[kind].label,
            description: t.settings.notifications.kinds[kind].description
          })
        } satisfies SettingDefinition
      ])
    ),
    completionSound: {
      subpage: 'sounds',
      keywords: ['sound', 'chime', 'ding', 'audio', 'done', 'complete', 'mute'],
      copy: t => ({
        label: t.settings.notifications.completionSoundTitle,
        description: t.settings.notifications.completionSoundDesc
      })
    }
  },
  sessions: {
    autoArchive: {
      subpage: 'archived',
      keywords: ['archive', 'stale', 'cleanup', 'old chats', 'days', 'after'],
      copy: t => ({ label: t.settings.sessions.autoArchiveTitle, description: t.settings.sessions.autoArchiveDesc })
    }
  },
  gateway: {
    connectionMode: {
      subpage: 'connection',
      keywords: ['gateway', 'connection', 'local', 'cloud', 'remote', 'ssh', 'url', 'token', 'host', 'port', 'key'],
      copy: t => ({ label: t.settings.gateway.modeTitle, description: t.settings.gateway.intro })
    },
    keychainEncryption: {
      subpage: 'connection',
      keywords: ['keychain', 'encrypt', 'secrets', 'secure storage', 'plain text'],
      copy: t => ({
        label: t.settings.gateway.keychainEncryptionTitle,
        description: t.settings.gateway.keychainEncryptionDesc
      })
    },
    diagnostics: {
      subpage: 'connection',
      keywords: ['diagnostics', 'logs', 'debug', 'report', 'troubleshoot'],
      copy: t => ({ label: t.settings.gateway.diagnostics, description: t.settings.gateway.diagnosticsDesc })
    }
  },
  about: {
    updates: {
      subpage: 'updates',
      keywords: ['update', 'auto update', 'download', 'release', 'version'],
      copy: t => ({ label: t.settings.about.updates })
    }
  }
} as const satisfies Record<string, Record<string, SettingDefinition>>

export type ManifestViewKey = keyof typeof SETTINGS_MANIFEST

/** Manifest keys that are config sections route as `config:<key>`; the rest are views of their own. */
const CONFIG_SECTION_KEYS: readonly ManifestViewKey[] = ['appearance', 'chat', 'advanced']

export const manifestView = (key: ManifestViewKey): SettingsView =>
  CONFIG_SECTION_KEYS.includes(key) ? `config:${key}` : (key as SettingsView)

const manifestKeyForView = (view: SettingsView): ManifestViewKey | undefined => {
  const key = view.startsWith('config:') ? view.slice('config:'.length) : view

  return Object.hasOwn(SETTINGS_MANIFEST, key) ? (key as ManifestViewKey) : undefined
}

const kebab = (key: string) => key.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)

/** The deep-link id of a manifest row: `settingId('appearance', 'tips') === 'appearance.tips'`. */
export const settingId = (view: ManifestViewKey, key: string) => `${view}.${kebab(key)}`

/** Notification kinds are generated from the store's kind list, so their ids are looked up by kind. */
export const notificationKindSettingId = (kind: (typeof NATIVE_NOTIFICATION_KINDS)[number]) =>
  settingId('notifications', `kind-${kind}`)

type SettingIds = { readonly [V in ManifestViewKey]: { readonly [K in keyof (typeof SETTINGS_MANIFEST)[V]]: string } }

/** `SETTING_IDS.appearance.tips === 'appearance.tips'` — the id a row carries and a palette hit targets. */
export const SETTING_IDS = Object.fromEntries(
  (Object.keys(SETTINGS_MANIFEST) as ManifestViewKey[]).map(view => [
    view,
    Object.fromEntries(Object.keys(SETTINGS_MANIFEST[view]).map(key => [key, settingId(view, key)]))
  ])
) as SettingIds

/** The DOM id a settings row carries so a deep link can scroll to and flash it. */
export const settingElementId = (id: string) => `setting-field-${id}`

const definitions = (view: ManifestViewKey): Record<string, SettingDefinition> => SETTINGS_MANIFEST[view]

/** The manifest row behind a deep-link id on a view, if the view owns one. */
export function settingDefinition(view: SettingsView, id: string): SettingDefinition | undefined {
  const key = manifestKeyForView(view)

  if (!key) {
    return undefined
  }

  const [settingKey] = Object.entries(SETTING_IDS[key]).find(([, candidate]) => candidate === id) ?? []

  return settingKey ? definitions(key)[settingKey] : undefined
}

export interface SettingSearchTarget extends SettingCopy {
  id: string
  keywords: string[]
  view: SettingsView
}

/** Every manifest row the command palette can land on right now, copy resolved. */
export function settingSearchTargets(t: Translations): SettingSearchTarget[] {
  return (Object.keys(SETTINGS_MANIFEST) as ManifestViewKey[]).flatMap(viewKey =>
    Object.entries(definitions(viewKey))
      .filter(([, setting]) => setting.available?.() ?? true)
      .map(([key, setting]) => ({
        id: SETTING_IDS[viewKey][key as keyof (typeof SETTING_IDS)[typeof viewKey]],
        keywords: [...setting.keywords],
        view: manifestView(viewKey),
        ...setting.copy(t)
      }))
  )
}
