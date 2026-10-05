import { Switch } from '@/components/ui/switch'
import type { HermesConfigRecord } from '@/types/hermes'
import { getNested } from './helpers'

const CONSENT_KEY = 'auxiliary.background_review.enabled'

interface ModelReviewConsentProps {
  task: string
  label: string
  config?: HermesConfigRecord
  applying: boolean
  onChange(key: string, value: boolean): Promise<void>
}

export function ModelReviewConsent({ task, label, config, applying, onChange }: ModelReviewConsentProps) {
  if (task !== 'background_review') return null
  return (
    <Switch
      aria-label={label}
      checked={getNested(config ?? {}, CONSENT_KEY) === true}
      disabled={!config || applying}
      onCheckedChange={value => void onChange(CONSENT_KEY, value)}
    />
  )
}
