import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { setInterfaceMode } from '@/store/interface-mode'

/** Simple mode's Settings rail footer: the one door back to every hidden page. */
export function SimpleSettingsFooter() {
  const { t } = useI18n()

  return (
    <Button
      onClick={() => {
        triggerHaptic('selection')
        setInterfaceMode('advanced')
      }}
      size="xs"
      variant="text"
    >
      {t.interfaceMode.switchToAdvanced}
    </Button>
  )
}
