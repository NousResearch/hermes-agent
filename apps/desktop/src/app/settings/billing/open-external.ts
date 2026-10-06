import { openExternalLink } from '@/lib/external-link'

// Optional-arg convenience over the canonical opener — the billing rows pass
// possibly-undefined URLs straight through from their view models.
export function openExternal(url?: string) {
  if (url) {
    openExternalLink(url)
  }
}
