import defaultPose from '@/components/pet/assets/hud/hermes-agent-default-cane.png'
import garyPose from '@/components/pet/assets/hud/hermes-agent-gary.webp'
import jarvisPose from '@/components/pet/assets/hud/hermes-agent-jarvis.webp'
import repokeeperPose from '@/components/pet/assets/hud/hermes-agent-repokeeper.webp'
import sabiskaPose from '@/components/pet/assets/hud/hermes-agent-sabiska.webp'
import warrenPose from '@/components/pet/assets/hud/hermes-agent-warren.webp'

export interface QuickEntryAgentVisual {
  accent: string
  glow: string
  pose: string
  role: string
}

const DEFAULT_VISUAL: QuickEntryAgentVisual = {
  accent: '#b7ff2a',
  glow: 'rgba(171, 63, 255, 0.38)',
  pose: defaultPose,
  role: 'Hermes commander'
}

const PROFILE_VISUALS: Record<string, QuickEntryAgentVisual> = {
  default: DEFAULT_VISUAL,
  gary: {
    accent: '#ff8a24',
    glow: 'rgba(255, 48, 194, 0.4)',
    pose: garyPose,
    role: 'Creative and growth'
  },
  jarvis: {
    accent: '#35e8ff',
    glow: 'rgba(22, 137, 255, 0.42)',
    pose: jarvisPose,
    role: 'CTO and architecture'
  },
  repokeeper: {
    accent: '#60ffd0',
    glow: 'rgba(69, 255, 178, 0.34)',
    pose: repokeeperPose,
    role: 'Repository caretaker'
  },
  sabiska: {
    accent: '#c67cff',
    glow: 'rgba(55, 229, 255, 0.36)',
    pose: sabiskaPose,
    role: 'Research and evidence'
  },
  warren: {
    accent: '#ffd54c',
    glow: 'rgba(25, 220, 126, 0.35)',
    pose: warrenPose,
    role: 'Finance and risk'
  }
}

export function quickEntryAgentVisual(profile?: null | string): QuickEntryAgentVisual {
  return PROFILE_VISUALS[profile?.trim().toLowerCase() || 'default'] ?? DEFAULT_VISUAL
}
