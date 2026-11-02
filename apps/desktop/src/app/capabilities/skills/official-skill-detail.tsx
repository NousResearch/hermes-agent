import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { CountSkeleton } from '@/components/ui/skeleton'
import { previewSkillHub, type ProfileScope, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import { Loader2 } from '@/lib/icons'
import type { OfficialSkillInfo } from '@/types/hermes'

import { PanelPill } from '../../overlays/panel'
import { asText, prettyName } from '../../settings/helpers'
import { DetailHeader } from '../primitives'

import { parseFrontmatter } from './frontmatter'

// Detail pane for a not-yet-installed catalog skill.
export function OfficialSkillDetail({
  installing,
  onInstall,
  profile,
  skill
}: {
  installing: boolean
  onInstall: () => void
  profile?: ProfileScope
  skill: OfficialSkillInfo
}) {
  const { t } = useI18n()

  const previewQuery = useQuery({
    queryKey: ['official-skill-preview', skill.identifier, profileScopeKey(profile)],
    queryFn: () => previewSkillHub(skill.identifier, profile),
    staleTime: 5 * 60_000,
    retry: false
  })

  const parsed = useMemo(
    () => (previewQuery.data?.skill_md ? parseFrontmatter(previewQuery.data.skill_md) : null),
    [previewQuery.data]
  )

  return (
    <>
      <DetailHeader
        description={asText(skill.description) || t.skills.noDescription}
        pills={
          <>
            <PanelPill>{prettyName(skill.category)}</PanelPill>
            <PanelPill tone="muted">{t.skills.officialPill}</PanelPill>
          </>
        }
        title={skill.name}
      />
      <div className="flex items-center gap-2">
        <Button disabled={installing} onClick={onInstall} size="xs" variant="textStrong">
          {installing && <Loader2 className="size-3 animate-spin" />}
          {installing ? t.skills.hub.installing : t.skills.hub.install}
        </Button>
      </div>
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
      {previewQuery.isLoading ? (
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
