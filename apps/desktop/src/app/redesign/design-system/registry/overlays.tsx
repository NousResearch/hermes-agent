import { useState } from 'react'

import { ActionsContextMenu, ActionsMenu, type MenuKit } from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Field } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { Tip } from '@/components/ui/tooltip'
import { Plus } from '@/lib/icons'

import type { GalleryEntry } from './types'

function DialogDemo() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="secondary">Rename agent</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Rename agent</DialogTitle>
          <DialogDescription>The new name shows everywhere this agent appears.</DialogDescription>
        </DialogHeader>
        <Field htmlFor="ds-dialog-name" label="Name">
          <Input defaultValue="research" id="ds-dialog-name" />
        </Field>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="secondary">Cancel</Button>
          </DialogClose>
          <DialogClose asChild>
            <Button>Save</Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function ConfirmDemo() {
  const [open, setOpen] = useState(false)

  return (
    <>
      <Button onClick={() => setOpen(true)} variant="destructiveOutline">
        Delete routine
      </Button>
      <ConfirmDialog
        cancelLabel="Keep it"
        confirmLabel="Delete"
        description="Its runs stay in the agent's history."
        destructive
        onClose={() => setOpen(false)}
        onConfirm={() => setOpen(false)}
        open={open}
        title="Delete “Morning digest”?"
      />
    </>
  )
}

function DropdownDemo() {
  const [compact, setCompact] = useState(true)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="secondary">Open menu</Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        <DropdownMenuLabel>View</DropdownMenuLabel>
        <DropdownMenuCheckboxItem checked={compact} onCheckedChange={setCompact}>
          Compact rows
        </DropdownMenuCheckboxItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem>
          New chat
          <DropdownMenuShortcut>⌘N</DropdownMenuShortcut>
        </DropdownMenuItem>
        <DropdownMenuItem disabled>Export</DropdownMenuItem>
        <DropdownMenuItem variant="destructive">Delete</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

const actionItems = (kit: MenuKit) => (
  <>
    <kit.Item>Rename</kit.Item>
    <kit.Item>Duplicate</kit.Item>
    <kit.Separator />
    <kit.Item variant="destructive">Delete</kit.Item>
  </>
)

export const OVERLAYS: readonly GalleryEntry[] = [
  {
    title: 'Dialog',
    source: 'components/ui/dialog.tsx',
    use: 'A short task that needs the whole attention: a title, a body and footer actions.',
    group: 'overlays',
    examples: [{ label: 'with a field', render: () => <DialogDemo /> }]
  },
  {
    title: 'ConfirmDialog',
    source: 'components/ui/confirm-dialog.tsx',
    use: 'Ask before something that cannot be undone; destructive focuses the safe choice.',
    group: 'overlays',
    examples: [{ label: 'destructive', render: () => <ConfirmDemo /> }]
  },
  {
    title: 'Sheet',
    source: 'components/ui/sheet.tsx',
    use: 'A panel from a window edge for a secondary task.',
    group: 'overlays',
    examples: [
      {
        label: 'side right',
        render: () => (
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="secondary">Open sheet</Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>Details</SheetTitle>
                <SheetDescription>Secondary information about the selected item.</SheetDescription>
              </SheetHeader>
            </SheetContent>
          </Sheet>
        )
      }
    ]
  },
  {
    title: 'Popover',
    source: 'components/ui/popover.tsx',
    use: 'Something small anchored to its trigger: a picker, a preview, a short form.',
    group: 'overlays',
    examples: [
      {
        label: 'default',
        render: () => (
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="secondary">Open popover</Button>
            </PopoverTrigger>
            <PopoverContent>Popovers float on the floating surface, stroke and shadow.</PopoverContent>
          </Popover>
        )
      }
    ]
  },
  {
    title: 'Tip',
    source: 'components/ui/tooltip.tsx',
    use: 'Name an unlabelled control on hover or keyboard focus; never a native title.',
    group: 'overlays',
    examples: [
      {
        label: 'on an icon button',
        render: () => (
          <Tip label="New chat">
            <Button aria-label="New chat" size="icon-sm" variant="secondary">
              <Plus />
            </Button>
          </Tip>
        )
      }
    ]
  },
  {
    title: 'DropdownMenu',
    source: 'components/ui/dropdown-menu.tsx',
    use: 'Actions and options behind a trigger: labels, checks, shortcuts, separators.',
    group: 'overlays',
    covers: ['components/ui/menu.ts'],
    examples: [{ label: 'items, check, shortcut, destructive', render: () => <DropdownDemo /> }]
  },
  {
    title: 'ActionsMenu',
    source: 'components/ui/actions-menu.tsx',
    use: "A row's ⋯ menu and its right-click menu from one item list, so they never drift.",
    group: 'overlays',
    covers: ['components/ui/context-menu.tsx'],
    examples: [
      {
        label: 'kebab',
        render: () => (
          <ActionsMenu ariaLabel="Agent actions" items={actionItems}>
            <Button aria-label="Agent actions" size="icon-sm" variant="ghost">
              <Codicon name="ellipsis" />
            </Button>
          </ActionsMenu>
        )
      },
      {
        label: 'right-click (context-menu.tsx)',
        render: () => (
          <ActionsContextMenu ariaLabel="Agent actions" items={actionItems}>
            <div className="ds-context-target">Right-click here</div>
          </ActionsContextMenu>
        )
      }
    ]
  }
]
