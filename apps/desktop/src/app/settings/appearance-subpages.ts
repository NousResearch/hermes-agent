import type { Translations } from '@/i18n'
import { TRANSLUCENCY_SUPPORTED } from '@/store/translucency'

export const APPEARANCE_SUBPAGES = [
  { id: 'general', labelKey: 'appearanceGeneral' },
  { id: 'theme', labelKey: 'appearanceTheme' },
  { id: 'typography', labelKey: 'appearanceTypography' },
  { id: 'window-layout', labelKey: 'appearanceWindowLayout' },
  { id: 'chat-display', labelKey: 'appearanceChatDisplay' },
  { id: 'pet', labelKey: 'appearancePet' }
] as const

export type AppearanceSubpageId = (typeof APPEARANCE_SUBPAGES)[number]['id']

interface AppearanceSettingCopy {
  description?: string
  label: string
}

interface AppearanceSetting {
  /** Rows that only exist on some platforms stay out of search, where a hit would scroll to nothing. */
  available?: () => boolean
  copy: (t: Translations) => AppearanceSettingCopy
  keywords: readonly string[]
  subpage: AppearanceSubpageId
}

type AppearanceCopy = Translations['settings']['appearance']
type TitledKey = { [K in keyof AppearanceCopy]: K extends `${infer Base}Title` ? Base : never }[keyof AppearanceCopy]

// Most rows read `<key>Title` / `<key>Desc` straight from the appearance
// catalog; a Desc that is a formatter (uiScale) is left to the row itself.
const appearanceCopy =
  <K extends TitledKey>(key: K) =>
  (t: Translations): AppearanceSettingCopy => {
    const catalog: Record<string, unknown> = t.settings.appearance
    const description = catalog[`${key}Desc`]

    return {
      label: t.settings.appearance[`${key}Title`],
      description: typeof description === 'string' ? description : undefined
    }
  }

/**
 * The manifest of every hand-built Appearance row. A key becomes the row's
 * deep-link id (`appearance.<kebab-case>`), its subpage routes that id, and its
 * copy + keywords feed the command palette — so a row cannot exist without
 * being searchable, and search cannot point at a row that is not there.
 * Schema-driven rows (`desktop.font_family`) are indexed from the schema and
 * only need a subpage here.
 */
export const APPEARANCE_SETTINGS = {
  language: {
    subpage: 'general',
    keywords: ['locale'],
    copy: t => ({ label: t.language.label, description: t.language.description })
  },
  introSplash: {
    subpage: 'general',
    keywords: ['splash', 'wordmark', 'empty chat', 'new chat'],
    copy: appearanceCopy('introSplash')
  },
  resumeLastSession: {
    subpage: 'general',
    keywords: ['resume', 'reopen', 'launch', 'startup', 'last chat', 'session'],
    copy: appearanceCopy('resumeLastSession')
  },
  tips: {
    subpage: 'general',
    keywords: ['tips', 'hints', 'coach marks', 'onboarding', 'help'],
    copy: appearanceCopy('tips')
  },
  tours: {
    subpage: 'general',
    keywords: ['tour', 'walkthrough', 'guide', 'onboarding', 'help'],
    copy: appearanceCopy('tours')
  },
  theme: { subpage: 'theme', keywords: ['color mode', 'skin', 'light', 'dark'], copy: appearanceCopy('theme') },
  uiScale: { subpage: 'typography', keywords: ['zoom', 'size'], copy: appearanceCopy('uiScale') },
  interfaceMode: {
    subpage: 'window-layout',
    keywords: ['simple', 'advanced', 'mode', 'interface', 'chrome', 'minimal', 'focus'],
    copy: t => ({ label: t.interfaceMode.title, description: t.interfaceMode.hint })
  },
  sessionDensity: {
    subpage: 'window-layout',
    keywords: ['sidebar', 'sessions', 'compact', 'comfortable', 'density'],
    copy: appearanceCopy('sessionDensity')
  },
  tabStrip: { subpage: 'window-layout', keywords: ['tabs', 'tab bar', 'strip'], copy: appearanceCopy('tabStrip') },
  appActions: {
    subpage: 'window-layout',
    keywords: ['titlebar', 'settings gear', 'layout', 'HUD', 'left', 'right', 'tabs'],
    copy: appearanceCopy('appActions')
  },
  minimizeToTray: {
    subpage: 'window-layout',
    keywords: ['tray', 'background', 'minimize', 'dock', 'taskbar', 'menu bar'],
    available: () => Boolean(window.hermesDesktop?.minimizeToTray),
    copy: t => ({ label: t.settings.config.minimizeToTrayTitle, description: t.settings.config.minimizeToTrayDesc })
  },
  translucency: {
    subpage: 'window-layout',
    keywords: ['opacity', 'transparent', 'glass', 'blur'],
    available: () => TRANSLUCENCY_SUPPORTED,
    copy: appearanceCopy('translucency')
  },
  backdrop: { subpage: 'window-layout', keywords: ['background', 'blur'], copy: appearanceCopy('backdrop') },
  composerPopout: {
    subpage: 'window-layout',
    keywords: ['composer', 'floating', 'drag', 'popout', 'dock', 'input'],
    copy: appearanceCopy('composerPopout')
  },
  userBubble: {
    subpage: 'chat-display',
    keywords: ['opacity', 'transparent', 'message', 'bubble'],
    copy: appearanceCopy('userBubble')
  },
  textDirection: {
    subpage: 'chat-display',
    keywords: ['rtl', 'ltr', 'right to left', 'left to right', 'bidi', 'arabic', 'hebrew', 'persian', 'align'],
    copy: appearanceCopy('textDirection')
  },
  hideThreadTimeline: {
    subpage: 'chat-display',
    keywords: ['thread', 'conversation', 'timeline', 'bars', 'rail', 'navigation', 'hide'],
    copy: appearanceCopy('hideThreadTimeline')
  },
  reactions: {
    subpage: 'chat-display',
    keywords: ['emoji', 'tapback', 'react', 'reactions'],
    copy: appearanceCopy('reactions')
  },
  vibeHearts: {
    subpage: 'chat-display',
    keywords: ['hearts', 'vibe', 'celebrate', 'confetti', 'fun'],
    copy: appearanceCopy('vibeHearts')
  },
  toolView: { subpage: 'chat-display', keywords: ['tool display', 'technical'], copy: appearanceCopy('toolView') },
  hideCodeDiffs: {
    subpage: 'chat-display',
    keywords: ['code', 'diff', 'patch', 'file edits', 'inline', 'added', 'removed'],
    copy: appearanceCopy('hideCodeDiffs')
  },
  reasoningCollapsed: {
    subpage: 'chat-display',
    keywords: ['thinking', 'reasoning', 'collapse', 'expand', 'chain of thought'],
    copy: appearanceCopy('reasoningCollapsed')
  },
  embeds: { subpage: 'chat-display', keywords: ['external content', 'privacy'], copy: appearanceCopy('embeds') },
  pet: {
    subpage: 'pet',
    keywords: ['pet', 'mascot', 'petdex', 'companion', 'buddy'],
    copy: t => ({ label: t.settings.appearance.pet.chooseTitle, description: t.settings.appearance.pet.chooseDesc })
  }
} as const satisfies Record<string, AppearanceSetting>

export type AppearanceSettingKey = keyof typeof APPEARANCE_SETTINGS

const APPEARANCE_SETTING_KEYS = Object.keys(APPEARANCE_SETTINGS) as AppearanceSettingKey[]
const appearanceSetting = (key: AppearanceSettingKey): AppearanceSetting => APPEARANCE_SETTINGS[key]

const kebab = (key: string) => key.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)

export const APPEARANCE_SETTING_IDS = Object.fromEntries(
  APPEARANCE_SETTING_KEYS.map(key => [key, `appearance.${kebab(key)}`])
) as Readonly<Record<AppearanceSettingKey, string>>

const SETTING_SUBPAGES: Readonly<Record<string, AppearanceSubpageId>> = {
  // Schema-driven rows that render on an Appearance subpage.
  'desktop.font_family': 'typography',
  'terminal.font_family': 'typography',
  ...Object.fromEntries(
    APPEARANCE_SETTING_KEYS.map(key => [APPEARANCE_SETTING_IDS[key], appearanceSetting(key).subpage])
  )
}

export function appearanceSubpageForSetting(setting: string): AppearanceSubpageId | undefined {
  return Object.hasOwn(SETTING_SUBPAGES, setting) ? SETTING_SUBPAGES[setting] : undefined
}

/** The Appearance rows the command palette can land on right now, with their copy resolved. */
export function appearanceSearchTargets(t: Translations) {
  return APPEARANCE_SETTING_KEYS.map(key => [key, appearanceSetting(key)] as const)
    .filter(([, setting]) => setting.available?.() ?? true)
    .map(([key, setting]) => ({ id: APPEARANCE_SETTING_IDS[key], keywords: [...setting.keywords], ...setting.copy(t) }))
}
