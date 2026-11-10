import * as React from 'react'

import { cn } from '@/lib/utils'

function TextTabMeta({ className, ...props }: React.ComponentProps<'span'>) {
  return <span className={cn('text-[0.72em] font-normal text-(--ui-text-tertiary)', className)} {...props} />
}

interface TextTabProps extends React.ComponentProps<'button'> {
  active?: boolean
  /** `underline` (default) for chrome rows; `pill` for marketplace page tabs. */
  variant?: 'underline' | 'pill'
}

function TextTab({
  active = false,
  children,
  className,
  type = 'button',
  variant = 'underline',
  ...props
}: TextTabProps) {
  const pill = variant === 'pill'

  return (
    <button
      className={cn(
        'group/text-tab inline-flex items-center gap-1 font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring disabled:pointer-events-none disabled:opacity-50',
        pill
          ? cn(
              'h-8 rounded-full px-3.5 text-sm',
              active
                ? 'bg-[color-mix(in_srgb,var(--ui-blue)_16%,transparent)] text-[color:var(--ui-blue)]'
                : 'bg-transparent text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
            )
          : cn(
              'h-7 bg-transparent px-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary) hover:bg-transparent hover:text-foreground focus-visible:text-foreground',
              active && 'text-foreground'
            ),
        className
      )}
      data-active={active}
      data-variant={variant}
      type={type}
      {...props}
    >
      {React.Children.map(children, child =>
        React.isValidElement(child) && child.type === TextTabMeta ? (
          child
        ) : pill ? (
          <span>{child}</span>
        ) : (
          <span
            className={cn(
              'underline-offset-4 decoration-current/25',
              active ? 'underline' : 'group-hover/text-tab:underline'
            )}
          >
            {child}
          </span>
        )
      )}
    </button>
  )
}

export { TextTab, TextTabMeta }
