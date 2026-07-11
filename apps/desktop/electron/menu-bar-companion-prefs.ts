export const DEFAULT_MENU_BAR_COMPANION_ENABLED = false

export function menuBarCompanionEnabledFromConfig(config: unknown): boolean {
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    return DEFAULT_MENU_BAR_COMPANION_ENABLED
  }

  const enabled = (config as { enabled?: unknown }).enabled

  return typeof enabled === 'boolean' ? enabled : DEFAULT_MENU_BAR_COMPANION_ENABLED
}
