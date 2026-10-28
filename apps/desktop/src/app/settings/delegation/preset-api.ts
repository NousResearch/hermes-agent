import { capabilityScoped, type ProfileScope } from '@/api/client'

export interface PresetState {
  names: string[]
  active: string
}

export interface PresetAction {
  action: 'save' | 'use' | 'rename' | 'delete'
  name: string
  new_name?: string
}

// Plugin-owned semantics and transport. An absent/older/disabled plugin is 404.
export function presetRequest(scope: ProfileScope, action?: PresetAction): Promise<PresetState> {
  return window.hermesDesktop.api<PresetState>({
    ...capabilityScoped(scope),
    path: '/api/plugins/advanced-delegation/presets',
    ...(action ? { method: 'POST', body: action } : {})
  })
}
