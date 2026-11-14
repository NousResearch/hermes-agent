import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { triggerHaptic } from '@/lib/haptics'

/** Simple mode's way back to a folded chat sidebar or page panel (Figma 1061:7816): one floating
 *  button at the stage's top-leading corner, under the band, where the panel's edge was. Advanced
 *  keeps the band's sidebar button instead, so a window never shows both. The open destination's rail
 *  button (and ⌘B) folds and restores the same panel. */
export function PanelRevealButton({ label, onReveal }: { label: string; onReveal: () => void }) {
  return (
    <Tip label={<TipKeybindLabel actionId="view.toggleSidebar" text={label} />} placement="toolbar">
      <Button
        aria-expanded={false}
        aria-label={label}
        className="portal-panel-reveal"
        data-slot="panel-reveal"
        onClick={() => {
          triggerHaptic('tap')
          onReveal()
        }}
        size="icon"
        variant="ghost"
      >
        <Codicon name="layout-sidebar-left-off" />
      </Button>
    </Tip>
  )
}
