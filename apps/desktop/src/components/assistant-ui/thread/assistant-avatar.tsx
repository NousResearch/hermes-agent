import { useStore } from '@nanostores/react'
import { type FC, useMemo } from 'react'

import { $profileAvatarDataUrl } from '@/store/avatar'
import { cn } from '@/lib/utils'

interface AssistantAvatarProps {
  /** Optional extra class names for sizing/positioning overrides. */
  className?: string
  /**
   * Optional name used to derive the fallback letter and colour.
   * Defaults to '🤖' when empty and shows the bot emoji instead of a letter.
   */
  name?: string
}

/**
 * Deterministic gradient colour palette for avatar fallbacks.
 * Each entry is [from, to] for a Tailwind `bg-linear-to-br` gradient.
 */
const AVATAR_GRADIENTS: [string, string][] = [
  ['#D946EF', '#A21CAF'], // fuchsia
  ['#8B5CF6', '#6D28D9'], // violet
  ['#3B82F6', '#1D4ED8'], // blue
  ['#06B6D4', '#0891B2'], // cyan
  ['#10B981', '#047857'], // emerald
  ['#F59E0B', '#D97706'], // amber
  ['#EF4444', '#DC2626'], // red
  ['#EC4899', '#DB2777'], // pink
  ['#14B8A6', '#0F766E'], // teal
  ['#F97316', '#EA580C'], // orange
]

/** Pick a deterministic gradient pair from a character. */
function gradientForLetter(letter: string): [string, string] {
  const code = letter.charCodeAt(0) || 65 // 'A' default
  return AVATAR_GRADIENTS[code % AVATAR_GRADIENTS.length]
}

/**
 * A small circular avatar rendered next to assistant messages.
 * When no custom avatar is uploaded, shows a gradient circle with the
 * first letter of the agent name, or the 🤖 emoji when no name is set.
 */
export const AssistantAvatar: FC<AssistantAvatarProps> = ({ className, name }) => {
  const avatarDataUrl = useStore($profileAvatarDataUrl)

  // Determine fallback display: first letter of name, or 🤖 if no name
  const fallbackChar = useMemo(() => {
    if (name && name.trim()) {
      return name.trim().charAt(0).toUpperCase()
    }

    return '🤖'
  }, [name])

  // Deterministic gradient colours from the fallback character
  const [gradientFrom, gradientTo] = useMemo(
    () => gradientForLetter(fallbackChar),
    [fallbackChar]
  )

  if (!avatarDataUrl) {
    const isEmoji = fallbackChar === '🤖'

    return (
      <div
        className={cn(
          'flex size-7 shrink-0 items-center justify-center self-start',
          className
        )}
        aria-hidden="true"
      >
        <div
          className={cn(
            'flex size-full items-center justify-center rounded-full',
            'bg-linear-to-br from-[var(--avatar-grad-from)] to-[var(--avatar-grad-to)]',
            'shadow-sm ring-1 ring-inset ring-white/15'
          )}
          style={{
            '--avatar-grad-from': gradientFrom,
            '--avatar-grad-to': gradientTo
          } as React.CSSProperties}
        >
          <span
            className={cn(
              'select-none font-semibold text-white drop-shadow-sm',
              isEmoji ? 'text-[0.8125rem] leading-none' : 'text-[0.6875rem] leading-none'
            )}
          >
            {fallbackChar}
          </span>
        </div>
      </div>
    )
  }

  return (
    <div
      className={cn(
        'flex size-7 shrink-0 items-center justify-center self-start overflow-hidden rounded-full',
        className
      )}
      aria-hidden="true"
    >
      <img
        alt=""
        className="size-full object-cover"
        draggable={false}
        src={avatarDataUrl}
      />
    </div>
  )
}
