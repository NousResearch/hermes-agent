import { useCallback } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { pathLabel } from '@/lib/chat-runtime'
import { FolderOpen } from '@/lib/icons'
import { compactProjectPath } from '@/lib/project-dir'
import { cn } from '@/lib/utils'

export interface NewChatProjectPickerProps {
  branch?: string
  className?: string
  cwd: string
  disabled?: boolean
  onChangeCwd: (path: string) => void | Promise<void>
}

export function NewChatProjectPicker({ branch, className, cwd, disabled, onChangeCwd }: NewChatProjectPickerProps) {
  const { t } = useI18n()
  const copy = t.newChatProject
  const trimmed = cwd.trim()
  const hasCwd = trimmed.length > 0
  const branchLabel = branch?.trim() || ''

  const chooseFolder = useCallback(async () => {
    const selected = await window.hermesDesktop?.selectPaths({
      defaultPath: hasCwd ? trimmed : undefined,
      directories: true,
      multiple: false,
      title: copy.chooseTitle
    })

    if (selected?.[0]) {
      await onChangeCwd(selected[0])
    }
  }, [copy.chooseTitle, hasCwd, onChangeCwd, trimmed])

  return (
    <div
      className={cn(
        'pointer-events-auto w-full rounded-xl border border-[color-mix(in_srgb,var(--dt-composer-ring)_22%,var(--dt-input))] bg-[color-mix(in_srgb,var(--dt-card)_78%,transparent)] px-3 py-2 shadow-composer backdrop-blur-[0.5rem]',
        className
      )}
      data-slot="new-chat-project-picker"
    >
      <div className="flex items-start gap-2">
        <FolderOpen aria-hidden className="mt-0.5 size-4 shrink-0 text-(--ui-text-tertiary)" />
        <div className="min-w-0 flex-1">
          <p className="text-[0.6875rem] font-medium text-(--ui-text-secondary)">{copy.label}</p>
          <p className="mt-0.5 truncate font-mono text-[0.75rem] text-foreground" title={hasCwd ? trimmed : undefined}>
            {hasCwd ? compactProjectPath(trimmed) : copy.notSet}
          </p>
          {branchLabel ? (
            <p className="mt-0.5 text-[0.6875rem] text-(--ui-text-tertiary)">
              {copy.branchPrefix} {branchLabel}
            </p>
          ) : (
            <p className="mt-0.5 text-[0.6875rem] text-(--ui-text-tertiary)">{copy.hint}</p>
          )}
        </div>
        <Button
          className="h-7 shrink-0 gap-1 px-2 text-[0.75rem]"
          disabled={disabled}
          onClick={() => void chooseFolder()}
          type="button"
          variant="outline"
        >
          <Codicon name="folder-opened" size="0.8125rem" />
          <span>{hasCwd ? copy.change : copy.choose}</span>
        </Button>
      </div>
      {hasCwd ? (
        <p className="mt-1.5 pl-6 text-[0.6875rem] text-(--ui-text-quaternary)">
          {copy.workspaceName(pathLabel(trimmed))}
        </p>
      ) : null}
    </div>
  )
}