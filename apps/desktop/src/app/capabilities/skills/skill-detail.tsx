import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { CountSkeleton } from '@/components/ui/skeleton'
import { getSkillContent, type ProfileScope, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import type { SkillInfo } from '@/types/hermes'

import { PanelPill } from '../../overlays/panel'
import { asText, prettyName } from '../../settings/helpers'
import { DetailHeader } from '../primitives'

import { parseFrontmatter } from './frontmatter'
import { categoryFor } from './skills-data'

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
  // Only learned/local skills are the user's to rewrite or archive — bundled,
  // hub, and plugin skills are managed by their sources.
  const editable = skill.provenance === 'agent'

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
      <DetailHeader
        description={asText(skill.description) || t.skills.noDescription}
        pills={
          <>
            <PanelPill>{prettyName(categoryFor(skill))}</PanelPill>
            {skill.provenance && skill.provenance !== 'bundled' && (
              <PanelPill tone={skill.provenance === 'agent' ? 'good' : 'muted'}>
                {t.skills.provenance[skill.provenance]}
              </PanelPill>
            )}
          </>
        }
        title={skill.name}
      />
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
      {parsed && parsed.meta.length > 0 && (
        <div className="grid gap-1 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3">
          {parsed.meta.map(([key, value]) => (
            <div className="flex gap-2 text-[0.68rem] leading-4" key={key}>
              <span className="w-24 shrink-0 font-medium text-(--ui-text-tertiary)">{key}</span>
              <span className="min-w-0 whitespace-pre-wrap break-words text-(--ui-text-secondary)">{value}</span>
            </div>
          ))}
        </div>
      )}
      {contentQuery.isLoading ? (
        <CountSkeleton />
      ) : parsed ? (
        <pre
          className="overflow-auto whitespace-pre-wrap wrap-break-word rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3 font-mono text-[0.68rem] leading-relaxed"
          data-selectable-text="true"
        >
          {parsed.body.trim() || t.skills.noDescription}
        </pre>
      ) : null}
    </>
  )
}
