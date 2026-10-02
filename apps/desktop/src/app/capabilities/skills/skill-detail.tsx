import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { getSkillContent, type ProfileScope, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import type { SkillInfo } from '@/types/hermes'

import { parseFrontmatter } from './frontmatter'
import { isEditableProvenance } from './skill-provenance'
import { skillRelations } from './skill-relations'

interface SkillDetailProps {
  onArchive: () => void
  onEdit: () => void
  profile?: ProfileScope
  skill: SkillInfo
  skills?: SkillInfo[]
  onSelectSkill?: (name: string) => void
}

export function SkillDetail({ onArchive, onEdit, profile, skill, skills, onSelectSkill }: SkillDetailProps) {
  const { t } = useI18n()
  // Origin only, never mutability: external mounts stay editable in place —
  // see ./skill-provenance (commit 8c8fc6c1ec).
  const editable = isEditableProvenance(skill.provenance)

  // The FULL skill — frontmatter metadata + complete SKILL.md body — for any
  // provenance, scoped to the Capabilities profile selector.
  const contentQuery = useQuery({
    queryKey: ['skill-content', skill.name, profileScopeKey(profile)],
    queryFn: () => getSkillContent(skill.name, profile),
    staleTime: 60_000
  })
  const parsed = useMemo(
    () => (contentQuery.data ? parseFrontmatter(contentQuery.data.content) : null),
    [contentQuery.data]
  )
  const relations = useMemo(() => skillRelations(contentQuery.data?.content ?? ''), [contentQuery.data])
  const installed = relations.filter(name => skills?.some(row => row.name === name))
  const missing = relations.filter(name => !skills?.some(row => row.name === name))

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
        <pre
          className="overflow-auto whitespace-pre-wrap wrap-break-word font-mono text-[0.68rem] leading-relaxed text-(--ui-text-secondary)"
          data-selectable-text="true"
        >
          {parsed.body.trim() || t.skills.noDescription}
        </pre>
      ) : null}
      {installed.length > 0 && (
        <section className="space-y-2">
          <h3 className="text-xs text-(--ui-text-tertiary)">{t.skills.relatedSkills}</h3>
          <div className="flex flex-wrap gap-2">
            {installed.map(name => (
              <Button key={name} size="xs" variant="text" onClick={() => onSelectSkill?.(name)}>
                {name}
              </Button>
            ))}
          </div>
        </section>
      )}
      {missing.length > 0 && (
        <section className="space-y-2 text-xs text-(--ui-text-tertiary)">
          <h3>{t.skills.missingRelatedSkills}</h3>
          <div className="flex flex-wrap gap-2">
            {missing.map(name => (
              <span key={name}>{name}</span>
            ))}
          </div>
        </section>
      )}
    </>
  )
}
