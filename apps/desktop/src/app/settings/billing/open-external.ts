import { openExternalLink } from '@/lib/external-link'

// Optional-arg convenience over the canonical opener — the billing rows pass
// possibly-undefined URLs straight through from their view models.
export function openExternal(url?: string) {
  if (url) {
    openExternalLink(url)
  }
}

// Strip a trailing Unicode arrow (`↗`) from labels that render beside an
// `<ExternalLink />` icon so the button never shows two arrows side by side.
export function externalActionLabel(label: string): string {
  return label.replace(/\s*↗\s*$/, '')
}
