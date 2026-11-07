import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'

import { $skillsCommunityFeed } from './store'

/** Explicit on-demand fetch of the community registries feed (ClawHub,
 * skills.sh, …). Off by default: the Skills catalog loads the small
 * first-party snapshot, so the ~60 MB full feed downloads only when the user
 * asks for it. */
export function SkillsCommunityToggle() {
  const { t } = useI18n()
  const community = useStore($skillsCommunityFeed)

  return (
    <Button
      aria-pressed={community}
      onClick={() => $skillsCommunityFeed.set(!community)}
      size="xs"
      variant={community ? 'default' : 'outline'}
    >
      {t.catalog.showCommunity}
    </Button>
  )
}
