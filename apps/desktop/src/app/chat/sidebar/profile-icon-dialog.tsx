import { EmojiPicker } from 'frimousse'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'

interface ProfileIconDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  label: string
  hasIcon: boolean
  onSelect: (emoji: string) => void
  onClear: () => void
}

// Per-profile emoji picker. Uses frimousse (React-19 headless picker) rendering
// NATIVE unicode emojis — no CDN images. A Dialog (modal) is used rather than a
// Popover so it doesn't fight the rail square's existing color Popover anchor /
// focus-outside handling. Picking an emoji or clearing closes the dialog.
export function ProfileIconDialog({ open, onOpenChange, label, hasIcon, onSelect, onClear }: ProfileIconDialogProps) {
  const { t } = useI18n()
  const p = t.profiles

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-xs p-0">
        <DialogHeader className="px-4 pt-4">
          <DialogTitle>{p.chooseIconFor(label)}</DialogTitle>
        </DialogHeader>
        <EmojiPicker.Root
          className="isolate flex h-[22rem] w-full flex-col bg-transparent"
          onEmojiSelect={emoji => {
            onSelect(emoji.emoji)
            onOpenChange(false)
          }}
        >
          <EmojiPicker.Search
            className="mx-4 mb-2 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-control-background) px-2 py-1.5 text-sm outline-none"
            placeholder={p.searchEmoji}
          />
          <EmojiPicker.Viewport className="relative flex-1 overflow-y-auto px-2">
            <EmojiPicker.Loading className="grid h-full place-items-center text-xs text-(--ui-text-tertiary)">
              {t.common.loading}
            </EmojiPicker.Loading>
            <EmojiPicker.Empty className="grid h-full place-items-center text-xs text-(--ui-text-tertiary)">
              {p.noEmoji}
            </EmojiPicker.Empty>
            <EmojiPicker.List
              className="select-none pb-2"
              components={{
                CategoryHeader: ({ category, ...props }) => (
                  <div
                    className="bg-(--ui-bg-elevated) px-1 pb-1 pt-2 text-[0.6875rem] font-medium text-(--ui-text-tertiary)"
                    {...props}
                  >
                    {category.label}
                  </div>
                ),
                Emoji: ({ emoji, ...props }) => (
                  <button
                    className="flex size-8 items-center justify-center rounded-md text-xl data-[active]:bg-(--ui-control-hover-background)"
                    {...props}
                  >
                    {emoji.emoji}
                  </button>
                )
              }}
            />
          </EmojiPicker.Viewport>
        </EmojiPicker.Root>
        <DialogFooter className="px-4 pb-4">
          <Button disabled={!hasIcon} onClick={onClear} type="button" variant="ghost">
            {p.clearIcon}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
