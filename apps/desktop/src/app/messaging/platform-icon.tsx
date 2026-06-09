import {
  SiApple,
  SiBilibili,
  SiDiscord,
  SiGmail,
  SiHomeassistant,
  SiMatrix,
  SiMattermost,
  SiQq,
  SiSignal,
  SiTelegram,
  SiWechat,
  SiWhatsapp
} from '@icons-pack/react-simple-icons'
import { type ComponentType, type SVGProps, useEffect, useState } from 'react'

import { Globe, Link as LinkIcon, MessageSquareText } from '@/lib/icons'
import { cn } from '@/lib/utils'

/** sRGB relative luminance (0 = black, 1 = white). */
export function luminance(hex: string): number {
  const clean = hex.trim().replace(/^#/, '')

  if (!/^[0-9a-f]{6}$/i.test(clean)) {
    return 0.5
  }

  const [r, g, b] = [0, 2, 4].map(i => {
    const c = parseInt(clean.slice(i, i + 2), 16) / 255

    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  })

  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

/**
 * Lift very-dark brand colours so the icon glyph stays legible on dark
 * surfaces.  Returns the original colour unchanged when in light mode or
 * when the colour is already bright enough.
 */
export function ensureVisible(hex: string, isDark: boolean): string {
  if (!isDark) {
    return hex
  }

  const LUMINANCE_FLOOR = 0.12 // ~#4a4a4a — readable on near-black surfaces

  if (luminance(hex) >= LUMINANCE_FLOOR) {
    return hex
  }

  // Parse and lighten the RGB channels uniformly.
  const clean = hex.trim().replace(/^#/, '')

  if (!/^[0-9a-f]{6}$/i.test(clean)) {
    return hex
  }

  const channels = [0, 2, 4].map(i => parseInt(clean.slice(i, i + 2), 16))

  // Binary-search the lightening factor that brings luminance up to the floor.
  let lo = 0
  let hi = 1

  for (let i = 0; i < 16; i++) {
    const mid = (lo + hi) / 2
    const blended = channels.map(c => Math.round(c + (255 - c) * mid))
    const lum = luminance(`#${blended.map(c => c.toString(16).padStart(2, '0')).join('')}`)

    if (lum < LUMINANCE_FLOOR) {
      lo = mid
    } else {
      hi = mid
    }
  }

  const factor = hi

  return `#${channels.map(c => Math.round(c + (255 - c) * factor).toString(16).padStart(2, '0')).join('')}`
}

// We render simpleicons.org brand glyphs for platforms whose owners publish a
// usable mark (telegram, discord, matrix, ...). A few brands — Slack, Dingtalk,
// Feishu, WeCom — have been removed from Simple Icons at the brand owner's
// request, so we fall back to a colored letter monogram for those.
//
// `iconColor` is the brand's hex from simpleicons.org so we can paint each
// glyph in its native color on top of a soft tint. The fallback monogram uses
// the same hex to keep visual consistency.
type IconKind = 'brand' | 'generic'

interface PlatformIconSpec {
  Icon?: ComponentType<SVGProps<SVGSVGElement>>
  color: string
  kind: IconKind
  monogram?: string
}

const PLATFORM_ICONS: Record<string, PlatformIconSpec> = {
  telegram: { Icon: SiTelegram, color: '#26A5E4', kind: 'brand' },
  discord: { Icon: SiDiscord, color: '#5865F2', kind: 'brand' },
  // Slack removed from Simple Icons by Salesforce request — letter monogram.
  slack: { color: '#4A154B', kind: 'brand', monogram: 'S' },
  mattermost: { Icon: SiMattermost, color: '#0058CC', kind: 'brand' },
  matrix: { Icon: SiMatrix, color: '#000000', kind: 'brand' },
  signal: { Icon: SiSignal, color: '#3A76F0', kind: 'brand' },
  whatsapp: { Icon: SiWhatsapp, color: '#25D366', kind: 'brand' },
  bluebubbles: { Icon: SiApple, color: '#0BD318', kind: 'brand' },
  homeassistant: { Icon: SiHomeassistant, color: '#18BCF2', kind: 'brand' },
  email: { Icon: SiGmail, color: '#EA4335', kind: 'brand' },
  sms: { Icon: MessageSquareText, color: '#F43F5E', kind: 'generic' },
  webhook: { Icon: LinkIcon, color: '#71717A', kind: 'generic' },
  api_server: { Icon: Globe, color: '#64748B', kind: 'generic' },
  weixin: { Icon: SiWechat, color: '#07C160', kind: 'brand' },
  qqbot: { Icon: SiQq, color: '#EB1923', kind: 'brand' },
  yuanbao: { Icon: SiBilibili, color: '#FB7299', kind: 'brand' }
}

interface PlatformAvatarProps {
  platformId: string
  platformName: string
  className?: string
}

/** Read the `.dark` class set by ThemeProvider on <html>. */
function useIsDark(): boolean {
  const [isDark, setIsDark] = useState(() =>
    typeof document !== 'undefined' && document.documentElement.classList.contains('dark')
  )

  useEffect(() => {
    if (typeof document === 'undefined') {
      return
    }

    const el = document.documentElement
    const observer = new MutationObserver(() => setIsDark(el.classList.contains('dark')))

    observer.observe(el, { attributes: true, attributeFilter: ['class'] })
    setIsDark(el.classList.contains('dark'))

    return () => observer.disconnect()
  }, [])

  return isDark
}

export function PlatformAvatar({ className, platformId, platformName }: PlatformAvatarProps) {
  const spec = PLATFORM_ICONS[platformId]
  const isDark = useIsDark()

  const baseClass = cn(
    'inline-grid size-6 shrink-0 place-items-center rounded-md text-[length:var(--conversation-caption-font-size)] font-medium',
    className
  )

  if (!spec) {
    return (
      <span aria-hidden="true" className={cn(baseClass, 'bg-(--ui-bg-tertiary) text-(--ui-text-tertiary)')}>
        {platformName.charAt(0).toUpperCase()}
      </span>
    )
  }

  const { Icon } = spec
  const color = ensureVisible(spec.color, isDark)

  return (
    <span
      aria-hidden="true"
      className={baseClass}
      style={{
        // 16% tint of the brand color so the glyph reads against any surface
        // without the avatar dominating the row.
        backgroundColor: `color-mix(in srgb, ${color} 16%, transparent)`,
        color
      }}
    >
      {Icon ? <Icon className="size-3.5" /> : spec.monogram || platformName.charAt(0).toUpperCase()}
    </span>
  )
}
