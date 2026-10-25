// A type alias, not an interface: callers index these labels as a string record.
export type SettingsNavTranslations = {
  providers: string
  providerAccounts: string
  providerApiKeys: string
  providerCustomEndpoints: string
  providerLocalModels: string
  gateway: string
  apiKeys: string
  keybinds: string
  keysTools: string
  keysSettings: string
  mcp: string
  archivedChats: string
  sessions: string
  about: string
  billing: string
  notifications: string
  vault: string
  plugins: string
}

export interface SettingsPageTranslations {
  back: string
  agent: string
  connections: string
  app: string
  actions: string
}
