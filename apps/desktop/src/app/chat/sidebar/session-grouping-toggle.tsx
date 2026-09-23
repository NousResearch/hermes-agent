import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { $sidebarGrouping, setSidebarGrouping } from '@/store/layout'
import { exitProjectScope } from '@/store/projects'

interface SessionGroupingToggleProps {
  className?: string
}

export function SessionGroupingToggle({ className }: SessionGroupingToggleProps) {
  const { t } = useI18n()
  const projectMode = useStore($sidebarGrouping) === 'project'
  const label = projectMode ? t.sidebar.showSessions : t.sidebar.showProjects

  return (
    <Tip label={label}>
      <Button
        aria-description={projectMode ? t.sidebar.groupAriaGrouped : t.sidebar.groupAriaUngrouped}
        aria-label={label}
        className={className}
        onClick={event => {
          event.stopPropagation()

          if (projectMode) {
            exitProjectScope()
          }

          setSidebarGrouping(projectMode ? 'date' : 'project')
        }}
        onPointerDown={event => event.preventDefault()}
        size="icon-xs"
        variant="ghost"
      >
        <Codicon name={projectMode ? 'list-unordered' : 'root-folder'} size="0.75rem" />
      </Button>
    </Tip>
  )
}
