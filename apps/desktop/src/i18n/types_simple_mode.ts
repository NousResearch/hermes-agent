// Simple-mode surfaces: the Settings interface-mode picker and the empty-chat
// starter ideas; `Translations` extends this.
export interface SimpleModeTranslations {
  interfaceMode: {
    title: string
    hint: string
    sessionNote: string
    switchToAdvanced: string
    simple: { label: string; description: string }
    advanced: { label: string; description: string }
  }
  chatStarters: {
    label: string
    intro: string
    email: { title: string; prompt: string }
    dinners: { title: string; prompt: string }
    summarize: { title: string; prompt: string }
    lookup: { title: string; prompt: string }
  }
}
