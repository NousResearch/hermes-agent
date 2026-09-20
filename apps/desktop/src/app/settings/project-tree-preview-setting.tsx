import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { ListRow } from '@/app/settings/primitives'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import {
  $projectTreePreviewLimit,
  PROJECT_TREE_PREVIEW_MAX,
  PROJECT_TREE_PREVIEW_MIN,
  setProjectTreePreviewLimit
} from '@/store/project-tree-preview-limit'

/** Settings → Advanced: how many session rows the sidebar previews under each
 *  project before the "show all sessions" toggle. Device-local (not
 *  profile-scoped): it only decides how much this window asks the backend for. */
export function ProjectTreePreviewSetting() {
  const { t } = useI18n()
  const limit = useStore($projectTreePreviewLimit)
  const [draft, setDraft] = useState(String(limit))

  useEffect(() => {
    setDraft(String(limit))
  }, [limit])

  const commit = () => {
    const parsed = Number(draft)

    if (!Number.isFinite(parsed) || parsed === limit) {
      setDraft(String(limit))

      return
    }

    setDraft(String(setProjectTreePreviewLimit(parsed)))
  }

  return (
    <ListRow
      action={
        <Input
          aria-label={t.settings.projectTreePreview.aria}
          className="w-20"
          inputMode="numeric"
          max={PROJECT_TREE_PREVIEW_MAX}
          min={PROJECT_TREE_PREVIEW_MIN}
          onBlur={commit}
          onChange={event => setDraft(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter') {
              event.currentTarget.blur()
            }
          }}
          type="number"
          value={draft}
        />
      }
      description="How many sessions each project previews in the sidebar before you expand it. Raise it to see more recent work at a glance; the project's “show all” toggle still loads the complete list."
      title={t.settings.projectTreePreview.title}
    />
  )
}
