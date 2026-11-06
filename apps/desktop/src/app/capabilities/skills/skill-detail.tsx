import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { getSkillContent, type ProfileScope, profileScopeKey, setSkillLocked } from '@/hermes'
import { useI18n } from '@/i18n'
import { Lock, LockOpen } from '@/lib/icons'
import { queryClient } from '@/lib/query-client'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import type { SkillInfo } from '@/types/hermes'

import { parseFrontmatter } from './frontmatter'
import { isEditableProvenance } from './skill-provenance'
import { skillsQueryKey } from './skills-data'

export function SkillDetail({
  onArchive,
  onEdit,
  profile,
  skill
}: {
  onArchive: () => void
  onEdit: () => void
  profile?: ProfileScope
  skill: SkillInfo
}) {
  const { t } = useI18n()
  // Origin only, never mutability: external mounts stay editable in place —
  // see ./skill-provenance (commit 8c8fc6c1ec).
  const editable = isEditableProvenance(skill.provenance)

  // The FULL skill — frontmatter metadata + complete SKILL.md body — for any
  // provenance, scoped to the Capabilities profile selector. The row list only
  // carries name/description; the pane shows the whole thing.
  const contentQuery = useQuery({
    queryKey: ['skill-content', skill.name, profileScopeKey(profile)],
    queryFn: () => getSkillContent(skill.name, profile),
    staleTime: 60_000
  })

  // Lock is a user declaration against AGENT edits — offered for every provenance
  // (bundled/hub/external can be locked too) and never blocks the user's own edits.
  const locked = Boolean(skill.locked)
  const [lockPending, setLockPending] = useState(false)

  const setLocked = (name: string, next: boolean) =>
    queryClient.setQueryData<SkillInfo[]>(skillsQueryKey(profile), current =>
      current?.map(row => (row.name === name ? { ...row, locked: next } : row))
    )

  const toggleLock = async () => {
    if (lockPending) {
      return
    }

    const { name } = skill
    const next = !locked
    setLockPending(true)
    await queryClient.cancelQueries({ queryKey: skillsQueryKey(profile), exact: true })
    setLocked(name, next)

    try {
      await setSkillLocked(name, next, profile)
    } catch (err) {
      setLocked(name, !next)
      notifyError(err, t.skills.failedToUpdate(name))
    } finally {
      setLockPending(false)
      void queryClient.invalidateQueries({ queryKey: skillsQueryKey(profile), exact: true })
    }
  }

  const parsed = useMemo(
    () => (contentQuery.data ? parseFrontmatter(contentQuery.data.content) : null),
    [contentQuery.data]
  )

  return (
    <>
      <div className="grid gap-1">
        <div className="flex items-center gap-2">
          {editable && (
            <>
              <Button onClick={onEdit} size="xs" variant="text">
                {t.skills.edit}
              </Button>
              <Button className="text-destructive hover:text-destructive" onClick={onArchive} size="xs" variant="text">
                {t.skills.archive}
              </Button>
            </>
          )}
          <Button
            aria-pressed={locked}
            className={cn(locked && 'text-foreground')}
            data-slot="skill-lock-toggle"
            disabled={lockPending}
            onClick={() => void toggleLock()}
            size="xs"
            variant="text"
          >
            {locked ? <Lock /> : <LockOpen />}
            {locked ? t.skills.locked : t.skills.lock}
          </Button>
        </div>
        <p className="text-xs text-(--ui-text-tertiary)">{locked ? t.skills.lockHint : t.skills.unlockedHint}</p>
      </div>
      {contentQuery.isLoading ? (
        <PageLoader className="h-40" label={t.skills.loading} />
      ) : parsed ? (
        <pre
          className="overflow-auto whitespace-pre-wrap wrap-break-word font-mono text-[0.68rem] leading-relaxed text-(--ui-text-secondary)"
          data-selectable-text="true"
        >
          {parsed.body.trim() || t.skills.noDescription}
        </pre>
      ) : null}
    </>
  )
}
