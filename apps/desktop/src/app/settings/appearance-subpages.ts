export const APPEARANCE_SUBPAGES = [
  { id: 'general', labelKey: 'appearanceGeneral' },
  { id: 'theme', labelKey: 'appearanceTheme' },
  { id: 'typography', labelKey: 'appearanceTypography' },
  { id: 'window-layout', labelKey: 'appearanceWindowLayout' },
  { id: 'chat-display', labelKey: 'appearanceChatDisplay', tier: 'advanced' },
  { id: 'pet', labelKey: 'appearancePet', tier: 'advanced' }
] as const

export type AppearanceSubpageId = (typeof APPEARANCE_SUBPAGES)[number]['id']
