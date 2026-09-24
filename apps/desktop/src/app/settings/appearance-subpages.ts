export const APPEARANCE_SUBPAGES = [
  { id: 'general', labelKey: 'appearanceGeneral' },
  { id: 'theme', labelKey: 'appearanceTheme' },
  { id: 'typography', labelKey: 'appearanceTypography' },
  { id: 'window-layout', labelKey: 'appearanceWindowLayout' },
  { id: 'chat-display', labelKey: 'appearanceChatDisplay' },
  { id: 'pet', labelKey: 'appearancePet' }
] as const

export type AppearanceSubpageId = (typeof APPEARANCE_SUBPAGES)[number]['id']

// Keep routing metadata independent of settings-search, which consumes it.
const SETTING_SUBPAGES: Readonly<Record<string, AppearanceSubpageId>> = {
  'appearance.app-actions': 'window-layout',
  'appearance.backdrop': 'window-layout',
  'appearance.embeds': 'chat-display',
  'appearance.hide-code-diffs': 'chat-display',
  'appearance.hide-thread-timeline': 'chat-display',
  'appearance.intro-splash': 'general',
  'appearance.language': 'general',
  'appearance.minimize-to-tray': 'window-layout',
  'appearance.pet': 'pet',
  'appearance.text-direction': 'chat-display',
  'appearance.theme': 'theme',
  'appearance.tool-view': 'chat-display',
  'appearance.interface-mode': 'window-layout',
  'appearance.translucency': 'window-layout',
  'appearance.ui-scale': 'typography',
  'appearance.user-bubble': 'chat-display',
  'desktop.font_family': 'typography',
  'terminal.font_family': 'typography'
}

export function appearanceSubpageForSetting(setting: string): AppearanceSubpageId | undefined {
  return Object.hasOwn(SETTING_SUBPAGES, setting) ? SETTING_SUBPAGES[setting] : undefined
}
