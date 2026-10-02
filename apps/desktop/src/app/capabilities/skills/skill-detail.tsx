import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { getSkillContent, type ProfileScope, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import type { SkillInfo } from '@/types/hermes'

import { parseFrontmatter } from './frontmatter'
import { isEditableProvenance } from './skill-provenance'

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

  const parsed = useMemo(
    () => (contentQuery.data ? parseFrontmatter(contentQuery.data.content) : null),
    [contentQuery.data]
  )

  return (
    <>
      {editable && (
        <div className="flex items-center gap-2">
          <Button onClick={onEdit} size="xs" variant="text">
            {t.skills.edit}
          </Button>
          <Button className="text-destructive hover:text-destructive" onClick={onArchive} size="xs" variant="text">
            {t.skills.archive}
          </Button>
        </div>
      )}
      {contentQuery.isLoading ? (
        <PageLoader className="h-40" label={t.skills.loading} />
      ) : parsed ? (
        <>
          {parsed.meta.length > 0 && (
            // Frontmatter metadata as key/value rows (the d5773bfc3ad detail-pane
            // contract) — keys/values come from the SKILL.md, so no i18n.
            <dl
              className="flex shrink-0 flex-col gap-0.5 overflow-auto font-mono text-[0.68rem] leading-relaxed"
              data-skill-frontmatter
            >
              {parsed.meta.map(([key, value]) => (
                <div className="flex gap-2" key={key}>
                  <dt className="w-28 shrink-0 truncate text-(--ui-text-tertiary)">{key}</dt>
                  <dd
                    className="min-w-0 flex-1 wrap-break-word whitespace-pre-wrap text-(--ui-text-secondary)"
                    data-selectable-text="true"
                  >
                    {value || '—'}
                  </dd>
                </div>
              ))}
            </dl>
          )}
          <pre
            className="overflow-auto whitespace-pre-wrap wrap-break-word font-mono text-[0.68rem] leading-relaxed text-(--ui-text-secondary)"
            data-selectable-text="true"
          >
            {parsed.body.trim() || t.skills.noDescription}
          </pre>
        </>
      ) : null}
    </>
  )
}
