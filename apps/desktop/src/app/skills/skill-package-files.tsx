import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'

import { Button } from '@/components/ui/button'
import { getSkillFiles } from '@/hermes'
import { useI18n } from '@/i18n'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import type { SkillInfo, SkillPackageFile, SkillPackageFileKind } from '@/types/hermes'

const SUPPORT_KINDS: SkillPackageFileKind[] = ['references', 'templates', 'scripts', 'assets']

export const SKILL_PACKAGE_FILES_QUERY_KEY = ['skill-package-files'] as const

export function SkillPackageFiles({
  onDelete,
  onOpen,
  skill
}: {
  onDelete: (file: SkillPackageFile) => void
  onOpen: (file: SkillPackageFile) => void
  skill: SkillInfo
}) {
  const { t } = useI18n()
  const profile = useStore($activeGatewayProfile)

  const { data, isError, isPending } = useQuery({
    queryKey: [...SKILL_PACKAGE_FILES_QUERY_KEY, normalizeProfileKey(profile), skill.name],
    queryFn: () => getSkillFiles(skill.name),
    staleTime: 0
  })

  const editable = skill.provenance === 'agent'
  const skillFile = data?.files.find(file => file.path === 'SKILL.md')

  return (
    <section aria-label={t.skills.files} className="space-y-2">
      <h4 className="text-[0.68rem] font-semibold uppercase tracking-wide text-muted-foreground/70">
        {t.skills.files}
      </h4>
      {isPending ? (
        <p className="text-xs text-muted-foreground">{t.skills.loadingFiles}</p>
      ) : isError || !data ? (
        <p className="text-xs text-destructive">{t.skills.filesLoadFailed}</p>
      ) : (
        <div className="space-y-3">
          {skillFile && <SkillFileRow editable={editable} file={skillFile} onDelete={onDelete} onOpen={onOpen} />}
          {SUPPORT_KINDS.map(kind => {
            const files = data.files.filter(file => file.kind === kind)

            if (files.length === 0) {
              return null
            }

            return (
              <div className="space-y-1" key={kind}>
                <div className="flex items-center justify-between text-[0.64rem] uppercase tracking-wide text-muted-foreground/55">
                  <span>{kind}</span>
                  <span>{files.length}</span>
                </div>
                {files.map(file => (
                  <SkillFileRow editable={editable} file={file} key={file.path} onDelete={onDelete} onOpen={onOpen} />
                ))}
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}

function SkillFileRow({
  editable,
  file,
  onDelete,
  onOpen
}: {
  editable: boolean
  file: SkillPackageFile
  onDelete: (file: SkillPackageFile) => void
  onOpen: (file: SkillPackageFile) => void
}) {
  const { t } = useI18n()

  return (
    <div className="group flex min-w-0 items-center gap-1 rounded-md bg-(--ui-bg-quinary) px-2 py-1">
      <button
        className="min-w-0 flex-1 cursor-pointer truncate text-left font-mono text-[0.68rem] text-(--ui-text-secondary) hover:text-foreground"
        onClick={() => onOpen(file)}
        title={file.path}
        type="button"
      >
        {file.path}
      </button>
      {file.is_binary && <span className="text-[0.6rem] text-muted-foreground/60">{t.skills.binary}</span>}
      {editable && file.path !== 'SKILL.md' && (
        <Button
          aria-label={t.skills.deleteFile(file.path)}
          className="text-destructive opacity-0 hover:text-destructive group-hover:opacity-100 focus:opacity-100"
          onClick={() => onDelete(file)}
          size="xs"
          variant="text"
        >
          {t.common.delete}
        </Button>
      )}
    </div>
  )
}
