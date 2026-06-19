import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Textarea } from '@/components/ui/textarea'

export function EditInstructionsDialog({
  onClose,
  onSave,
  open,
  projectTitle,
  value
}: {
  onClose: () => void
  onSave: (text: string) => void
  open: boolean
  projectTitle: string
  value: string
}) {
  const [text, setText] = useState(value)

  useEffect(() => {
    if (open) {
      setText(value)
    }
  }, [open, value])

  return (
    <Dialog onOpenChange={v => !v && onClose()} open={open}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Instructions</DialogTitle>
          <DialogDescription>
            Provide Hermes with relevant instructions and information for chats within {projectTitle}. This will work alongside your agent profile instructions.
          </DialogDescription>
        </DialogHeader>

        <Textarea
          className="min-h-48 resize-none font-mono text-xs leading-5"
          maxLength={8000}
          onChange={e => setText(e.target.value)}
          placeholder="Add project-specific instructions here..."
          rows={12}
          value={text}
        />
        <p className="text-right text-[0.66rem] text-muted-foreground">{text.length}/8000</p>

        <DialogFooter>
          <Button onClick={onClose} type="button" variant="ghost">
            Cancel
          </Button>
          <Button onClick={() => onSave(text)} type="button">
            Save Instructions
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
