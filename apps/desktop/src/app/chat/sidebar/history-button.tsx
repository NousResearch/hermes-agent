import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { openCommandPalettePage } from '@/store/command-palette'

export function SidebarHistoryButton() {
  const { t } = useI18n()

  return (
    <Button
      className="shrink-0 justify-start text-(--ui-text-secondary)"
      onClick={() => openCommandPalettePage('history')}
      size="sm"
      type="button"
      variant="ghost"
    >
      <Codicon name="history" />
      {t.keybinds.actions['session.history']}…
    </Button>
  )
}
