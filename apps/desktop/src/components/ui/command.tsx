import { Command as CommandPrimitive } from 'cmdk'
import * as React from 'react'

import { Codicon } from '@/components/ui/codicon'
import { usePointerQuiet } from '@/components/ui/keyboard-first'
import { menuItemClass } from '@/components/ui/menu'
import { SearchIcon } from '@/lib/icons'
import { cn } from '@/lib/utils'

type CommandVariant = 'default' | 'menu'

// `default` is the palette look (command palette, model/session pickers).
// `menu` is a picker list inside `<PopoverContent variant="menu">`: the parts
// read the DropdownMenu/Select rows, so a combobox looks like every other menu.
const CommandVariantContext = React.createContext<CommandVariant>('default')

function Command({
  className,
  variant = 'default',
  ...props
}: React.ComponentProps<typeof CommandPrimitive> & { variant?: CommandVariant }) {
  return (
    <CommandVariantContext.Provider value={variant}>
      <CommandPrimitive
        className={cn(
          'flex h-full w-full flex-col overflow-hidden text-popover-foreground',
          variant === 'menu' ? 'bg-transparent' : 'rounded-md bg-popover',
          className
        )}
        data-slot="command"
        {...props}
      />
    </CommandVariantContext.Provider>
  )
}

interface CommandInputProps extends React.ComponentProps<typeof CommandPrimitive.Input> {
  /** Inline trailing slot, rendered on the right of the search row. */
  right?: React.ReactNode
}

function CommandInput({ className, right, ...props }: CommandInputProps) {
  const variant = React.useContext(CommandVariantContext)

  if (variant === 'menu') {
    // Same row as DropdownMenuSearch + its hairline: no icon, no box.
    return (
      <div
        className="mb-1 flex items-center gap-2 border-b border-(--ui-stroke-tertiary) px-2 py-1.5"
        data-slot="command-input-wrapper"
      >
        <CommandPrimitive.Input
          className={cn(
            'h-4 w-full bg-transparent text-xs leading-none text-foreground outline-none placeholder:text-(--ui-text-tertiary) disabled:cursor-not-allowed disabled:opacity-50',
            className
          )}
          data-slot="command-input"
          spellCheck={false}
          {...props}
        />
        {right}
      </div>
    )
  }

  return (
    <div className="flex h-11 items-center gap-2 border-b border-border px-3" data-slot="command-input-wrapper">
      <SearchIcon className="size-4 shrink-0 text-muted-foreground" />
      <CommandPrimitive.Input
        className={cn(
          'flex h-10 w-full rounded-md bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50',
          className
        )}
        data-slot="command-input"
        {...props}
      />
      {right}
    </div>
  )
}

function CommandList({ className, ...props }: React.ComponentProps<typeof CommandPrimitive.List>) {
  const variant = React.useContext(CommandVariantContext)
  // cmdk selects on pointer-enter, so a list that opens under a parked cursor —
  // or re-flows under one as the query narrows — hands the selection to
  // whatever row slid beneath the mouse, and Enter commits THAT. Inert until
  // the pointer actually moves (see usePointerQuiet).
  const pointerQuiet = usePointerQuiet()

  return (
    <CommandPrimitive.List
      className={cn(
        'overflow-y-auto overflow-x-hidden',
        variant === 'menu' ? 'dt-portal-scrollbar max-h-72' : 'max-h-100',
        pointerQuiet && 'pointer-events-none',
        className
      )}
      data-slot="command-list"
      {...props}
    />
  )
}

function CommandEmpty({ className, ...props }: React.ComponentProps<typeof CommandPrimitive.Empty>) {
  const variant = React.useContext(CommandVariantContext)

  return (
    <CommandPrimitive.Empty
      className={cn(
        variant === 'menu'
          ? 'px-2 py-3 text-center text-xs text-(--ui-text-tertiary)'
          : 'py-6 text-center text-sm text-muted-foreground',
        className
      )}
      data-slot="command-empty"
      {...props}
    />
  )
}

function CommandGroup({ className, ...props }: React.ComponentProps<typeof CommandPrimitive.Group>) {
  const variant = React.useContext(CommandVariantContext)

  return (
    <CommandPrimitive.Group
      className={cn(
        'overflow-hidden text-foreground **:[[cmdk-group-heading]]:sticky **:[[cmdk-group-heading]]:top-0 **:[[cmdk-group-heading]]:z-10',
        variant === 'menu'
          ? // menuLabelClass, spelled out so Tailwind can see it.
            'p-0 **:[[cmdk-group-heading]]:bg-(--ui-bg-elevated) **:[[cmdk-group-heading]]:px-2 **:[[cmdk-group-heading]]:py-1 **:[[cmdk-group-heading]]:text-xs **:[[cmdk-group-heading]]:font-medium **:[[cmdk-group-heading]]:text-(--ui-text-tertiary)'
          : 'p-1 **:[[cmdk-group-heading]]:bg-popover **:[[cmdk-group-heading]]:px-2 **:[[cmdk-group-heading]]:py-1.5 **:[[cmdk-group-heading]]:text-xs **:[[cmdk-group-heading]]:font-medium **:[[cmdk-group-heading]]:text-muted-foreground',
        className
      )}
      data-slot="command-group"
      {...props}
    />
  )
}

function CommandSeparator({ className, ...props }: React.ComponentProps<typeof CommandPrimitive.Separator>) {
  const variant = React.useContext(CommandVariantContext)

  return (
    <CommandPrimitive.Separator
      className={cn('-mx-1 h-px', variant === 'menu' ? 'my-1 bg-(--ui-stroke-tertiary)' : 'bg-border', className)}
      data-slot="command-separator"
      {...props}
    />
  )
}

function CommandItem({ className, ...props }: React.ComponentProps<typeof CommandPrimitive.Item>) {
  const variant = React.useContext(CommandVariantContext)

  return (
    <CommandPrimitive.Item
      className={cn(
        variant === 'menu'
          ? cn(
              menuItemClass,
              'cursor-pointer data-[selected=true]:bg-(--ui-control-active-background) data-[selected=true]:text-foreground'
            )
          : 'relative flex cursor-default select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none data-[selected=true]:bg-accent data-[selected=true]:text-accent-foreground',
        'data-[disabled=true]:pointer-events-none data-[disabled=true]:opacity-50',
        className
      )}
      data-slot="command-item"
      {...props}
    />
  )
}

/** The trailing check a menu row wears when it's the current value — same glyph and spot as DropdownMenu/Select. */
function CommandItemCheck({ checked }: { checked: boolean }) {
  return checked ? <Codicon className="ml-auto pl-2 text-foreground" name="check" size="0.75rem" /> : null
}

function CommandShortcut({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span
      className={cn('ml-auto text-xs tracking-widest text-muted-foreground', className)}
      data-slot="command-shortcut"
      {...props}
    />
  )
}

export {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandItemCheck,
  CommandList,
  CommandSeparator,
  CommandShortcut
}
