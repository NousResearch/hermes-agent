/**
 * ARO DESIGN SYSTEM — ICON SET (desktop port)
 *
 * Self-contained inline SVGs ported from the Aro Workbench prototype
 * (src/workbench/components/Icons.tsx). No codicon dependency, no external
 * icon library — every glyph is inline `currentColor` line art that
 * inherits text colour, so the aro token classes tint them per theme.
 */

import type { SVGProps } from 'react'

import { cn } from '@/lib/utils'

type P = SVGProps<SVGSVGElement> & { size?: number }

const base = ({ size, ...p }: P) => ({
  width: size ?? 15,
  height: size ?? 15,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.7,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  ...p,
})

/* ------------------------------ core glyphs ------------------------------ */
export const IconPulse = (p: P) => (
  <svg {...base(p)}>
    <path d="M3 12h4l2.5-7 4 14 2.5-7h5" />
  </svg>
)
export const IconBolt = (p: P) => (
  <svg {...base(p)}>
    <path d="M13 2 4.5 13.5H11l-1 8.5 8.5-11.5H12l1-8.5Z" />
  </svg>
)
export const IconTerminal = (p: P) => (
  <svg {...base(p)}>
    <rect x="2.5" y="4" width="19" height="16" rx="2.5" />
    <path d="m7 10 2.5 2.5L7 15M12.5 15.5h4" />
  </svg>
)
export const IconLayers = (p: P) => (
  <svg {...base(p)}>
    <path d="m12 3 8.5 4.5L12 12 3.5 7.5 12 3Z" />
    <path d="m3.5 12 8.5 4.5 8.5-4.5M3.5 16.5 12 21l8.5-4.5" />
  </svg>
)
export const IconGit = (p: P) => (
  <svg {...base(p)}>
    <circle cx="6.5" cy="5.5" r="2.5" />
    <circle cx="6.5" cy="18.5" r="2.5" />
    <circle cx="17.5" cy="12" r="2.5" />
    <path d="M6.5 8v8M9 5.5h4.5a2 2 0 0 1 2 2v2M17.5 14.5V17a2 2 0 0 1-2 2H9" />
  </svg>
)
export const IconPalette = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 3a9 9 0 1 0 0 18c1.2 0 1.8-.9 1.8-1.8 0-1.9 1.2-2.7 3-2.7h1.4A2.8 2.8 0 0 0 21 13.7 9 9 0 0 0 12 3Z" />
    <circle cx="8" cy="10" r="1.1" fill="currentColor" stroke="none" />
    <circle cx="12" cy="7.5" r="1.1" fill="currentColor" stroke="none" />
    <circle cx="16" cy="10" r="1.1" fill="currentColor" stroke="none" />
  </svg>
)
export const IconMonitor = (p: P) => (
  <svg {...base(p)}>
    <rect x="2.5" y="4" width="19" height="12.5" rx="2" />
    <path d="M9 20h6M12 16.5V20" />
  </svg>
)
export const IconSearch = (p: P) => (
  <svg {...base(p)}>
    <circle cx="11" cy="11" r="6.5" />
    <path d="m16 16 4.5 4.5" />
  </svg>
)
export const IconPlus = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 5v14M5 12h14" />
  </svg>
)
export const IconChevron = (p: P) => (
  <svg {...base(p)}>
    <path d="m9 6 6 6-6 6" />
  </svg>
)
export const IconChevronDown = (p: P) => (
  <svg {...base(p)}>
    <path d="m6 9 6 6 6-6" />
  </svg>
)
export const IconCheck = (p: P) => (
  <svg {...base(p)}>
    <path d="m4.5 12.5 5 5 10-11" />
  </svg>
)
export const IconX = (p: P) => (
  <svg {...base(p)}>
    <path d="M6 6l12 12M18 6 6 18" />
  </svg>
)
export const IconFile = (p: P) => (
  <svg {...base(p)}>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z" />
    <path d="M13.5 3.5V8H19" />
  </svg>
)
export const IconFolder = (p: P) => (
  <svg {...base(p)}>
    <path d="M3.5 7.5A2 2 0 0 1 5.5 5.5h3.2l2 2.5h7.8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2v-10.5Z" />
  </svg>
)
export const IconSpark = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 3.5 13.8 9l5.5 1.8-5.5 1.8L12 18l-1.8-5.4L4.7 10.8 10.2 9 12 3.5Z" />
  </svg>
)
export const IconBrain = (p: P) => (
  <svg {...base(p)}>
    <path d="M9.5 4.5A2.5 2.5 0 0 0 7 7a2.5 2.5 0 0 0-2 4 2.6 2.6 0 0 0 .6 4.2A2.6 2.6 0 0 0 8.2 20c.9 0 1.3-.6 1.3-1.4V4.5Z" />
    <path d="M14.5 4.5A2.5 2.5 0 0 1 17 7a2.5 2.5 0 0 1 2 4 2.6 2.6 0 0 1-.6 4.2A2.6 2.6 0 0 1 15.8 20c-.9 0-1.3-.6-1.3-1.4V4.5Z" />
  </svg>
)
export const IconShield = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 3 5 5.8v5.4c0 4.2 2.9 8 7 9.8 4.1-1.8 7-5.6 7-9.8V5.8L12 3Z" />
    <path d="m9 12 2 2 4-4" />
  </svg>
)
export const IconClock = (p: P) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M12 7.5V12l3 1.8" />
  </svg>
)
export const IconCoin = (p: P) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M12 7.5v9M14.5 9.5c-.5-.9-1.4-1.3-2.5-1.3-1.4 0-2.5.7-2.5 1.9 0 2.6 5 1.2 5 3.8 0 1.2-1.1 1.9-2.5 1.9-1.1 0-2-.4-2.5-1.3" />
  </svg>
)
export const IconBranch = (p: P) => (
  <svg {...base(p)}>
    <circle cx="7" cy="5.5" r="2.2" />
    <circle cx="7" cy="18.5" r="2.2" />
    <circle cx="17" cy="8" r="2.2" />
    <path d="M7 7.7v8.6M14.9 9.2c-.6 2.2-2.6 3.3-5.4 3.6" />
  </svg>
)
export const IconPlay = (p: P) => (
  <svg {...base(p)}>
    <path d="M7 4.8 19 12 7 19.2V4.8Z" />
  </svg>
)
export const IconPause = (p: P) => (
  <svg {...base(p)}>
    <path d="M8 5v14M16 5v14" strokeWidth="2.4" />
  </svg>
)
export const IconStop = (p: P) => (
  <svg {...base(p)}>
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
)
export const IconUndo = (p: P) => (
  <svg {...base(p)}>
    <path d="M4 9h9a5.5 5.5 0 1 1 0 11H8" />
    <path d="M4 9l4-4M4 9l4 4" />
  </svg>
)
export const IconPlug = (p: P) => (
  <svg {...base(p)}>
    <path d="M9 3v5M15 3v5M6 8h12v3a6 6 0 0 1-6 6 6 6 0 0 1-6-6V8ZM12 17v4" />
  </svg>
)
export const IconGrid = (p: P) => (
  <svg {...base(p)}>
    <rect x="3.5" y="3.5" width="7" height="7" rx="1.5" />
    <rect x="13.5" y="3.5" width="7" height="7" rx="1.5" />
    <rect x="3.5" y="13.5" width="7" height="7" rx="1.5" />
    <rect x="13.5" y="13.5" width="7" height="7" rx="1.5" />
  </svg>
)
export const IconList = (p: P) => (
  <svg {...base(p)}>
    <path d="M8 6.5h13M8 12h13M8 17.5h13M3.5 6.5h.01M3.5 12h.01M3.5 17.5h.01" />
  </svg>
)
export const IconStar = (p: P) => (
  <svg {...base(p)}>
    <path d="m12 4 2.4 5 5.6.8-4 4 .9 5.6-4.9-2.7L7.1 19.4 8 13.8l-4-4L9.6 9 12 4Z" />
  </svg>
)
export const IconArrowUp = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 19V5M6 11l6-6 6 6" />
  </svg>
)
export const IconPaperclip = (p: P) => (
  <svg {...base(p)}>
    <path d="M20 11.5 12.4 19a4.5 4.5 0 0 1-6.4-6.4l7.6-7.5a3 3 0 0 1 4.3 4.3l-7.6 7.5a1.5 1.5 0 0 1-2.2-2.1l7-7" />
  </svg>
)
export const IconAt = (p: P) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="4" />
    <path d="M16 8v5a3 3 0 0 0 5-2 9 9 0 1 0-3.5 7.1" />
  </svg>
)
export const IconWarning = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 4 3 19.5h18L12 4Z" />
    <path d="M12 10v4M12 16.8h.01" />
  </svg>
)
export const IconRefresh = (p: P) => (
  <svg {...base(p)}>
    <path d="M20 12a8 8 0 1 1-2.6-5.9" />
    <path d="M20 4v4.5h-4.5" />
  </svg>
)
export const IconEye = (p: P) => (
  <svg {...base(p)}>
    <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
)
export const IconMcp = (p: P) => (
  <svg {...base(p)}>
    <rect x="3" y="13.5" width="6" height="6" rx="1.5" />
    <rect x="15" y="13.5" width="6" height="6" rx="1.5" />
    <rect x="9" y="3.5" width="6" height="6" rx="1.5" />
    <path d="M12 9.5V11M6 13.5v-1.2a1.3 1.3 0 0 1 1.3-1.3h9.4a1.3 1.3 0 0 1 1.3 1.3v1.2" />
  </svg>
)

export const IconGrip = (p: P) => (
  <svg {...base(p)}>
    <circle cx="9" cy="6" r="1.1" fill="currentColor" />
    <circle cx="15" cy="6" r="1.1" fill="currentColor" />
    <circle cx="9" cy="12" r="1.1" fill="currentColor" />
    <circle cx="15" cy="12" r="1.1" fill="currentColor" />
    <circle cx="9" cy="18" r="1.1" fill="currentColor" />
    <circle cx="15" cy="18" r="1.1" fill="currentColor" />
  </svg>
)
export const IconZap = (p: P) => (
  <svg {...base(p)}>
    <path d="M13.5 2.5 4.8 13.4a.6.6 0 0 0 .5 1h5.2l-1 7.1 8.7-10.9a.6.6 0 0 0-.5-1h-5.2l1-7.1Z" />
  </svg>
)
export const IconLayout = (p: P) => (
  <svg {...base(p)}>
    <rect x="3" y="4" width="18" height="16" rx="2.5" />
    <path d="M3 15.5h18M9.5 4v11.5M15.5 15.5V11" />
  </svg>
)
export const IconUser = (p: P) => (
  <svg {...base(p)}>
    <circle cx="12" cy="8.5" r="3.5" />
    <path d="M4.8 20a7.2 7.2 0 0 1 14.4 0" />
  </svg>
)
export const IconBell = (p: P) => (
  <svg {...base(p)}>
    <path d="M6.5 16V11a5.5 5.5 0 1 1 11 0v5l1.5 2h-14l1.5-2Z" />
    <path d="M10.2 20a1.9 1.9 0 0 0 3.6 0" />
  </svg>
)
export const IconMic = (p: P) => (
  <svg {...base(p)}>
    <rect x="9" y="3" width="6" height="12" rx="3" />
    <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3M9 21h6" />
  </svg>
)
export const IconWave = (p: P) => (
  <svg {...base(p)}>
    <path d="M3 10v4M7 6v12M11 3v18M15 7v10M19 9v6M23 11v2" />
  </svg>
)
export const IconHelp = (p: P) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M9.7 9.4a2.4 2.4 0 0 1 4.6.8c0 1.6-2.3 1.8-2.3 3.3" />
    <path d="M12 16.6h.01" />
  </svg>
)
export const IconGlobe = (p: P) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M3.6 12h16.8M12 3.5c2.2 2.4 3.3 5.3 3.3 8.5S14.2 18.1 12 20.5c-2.2-2.4-3.3-5.3-3.3-8.5S9.8 5.9 12 3.5Z" />
  </svg>
)
export const IconCalendar = (p: P) => (
  <svg {...base(p)}>
    <rect x="3.5" y="5" width="17" height="15.5" rx="2.5" />
    <path d="M3.5 10h17M8 3.5v3M16 3.5v3" />
  </svg>
)
export const IconMail = (p: P) => (
  <svg {...base(p)}>
    <rect x="2.8" y="5" width="18.4" height="14" rx="2.5" />
    <path d="m3.5 7.5 8.5 6 8.5-6" />
  </svg>
)
export const IconCopy = (p: P) => (
  <svg {...base(p)}>
    <rect x="9" y="9" width="11.5" height="11.5" rx="2.2" />
    <path d="M15 6.2A2.2 2.2 0 0 0 12.8 4H5.7A2.2 2.2 0 0 0 3.5 6.2v7.1A2.2 2.2 0 0 0 5.7 15.5" />
  </svg>
)
export const IconPencil = (p: P) => (
  <svg {...base(p)}>
    <path d="m14 5 5 5M4 20l4.2-.9L19.3 8a2.1 2.1 0 0 0-3-3L5.2 16.1 4 20Z" />
  </svg>
)
export const IconTrash = (p: P) => (
  <svg {...base(p)}>
    <path d="M4.5 7h15M9.5 7V5.2A1.7 1.7 0 0 1 11.2 3.5h1.6A1.7 1.7 0 0 1 14.5 5.2V7" />
    <path d="M6.5 7 7.4 20a1.6 1.6 0 0 0 1.6 1.5h6a1.6 1.6 0 0 0 1.6-1.5L17.5 7" />
    <path d="M10.5 11v6M13.5 11v6" />
  </svg>
)
export const IconCommand = (p: P) => (
  <svg {...base(p)}>
    <path d="M8.5 8.5h7v7h-7z" />
    <path d="M8.5 8.5V6.2a2.3 2.3 0 1 0-2.3 2.3h2.3m7 0V6.2a2.3 2.3 0 1 1 2.3 2.3h-2.3m0 7v2.3a2.3 2.3 0 1 0 2.3-2.3h-2.3m-7 0v2.3a2.3 2.3 0 1 1-2.3-2.3h2.3" />
  </svg>
)
export const IconExternal = (p: P) => (
  <svg {...base(p)}>
    <path d="M14 4h6v6M20 4l-8.5 8.5" />
    <path d="M18 14.5V18a2.5 2.5 0 0 1-2.5 2.5H6A2.5 2.5 0 0 1 3.5 18V8.5A2.5 2.5 0 0 1 6 6h3.5" />
  </svg>
)
export const IconInbox = (p: P) => (
  <svg {...base(p)}>
    <path d="M3.5 13.5 6 5.2A2 2 0 0 1 7.9 4h8.2A2 2 0 0 1 18 5.2l2.5 8.3V18a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2v-4.5Z" />
    <path d="M3.5 13.5H8l1.2 2.2h5.6l1.2-2.2h4.5" />
  </svg>
)
export const IconArchive = (p: P) => (
  <svg {...base(p)}>
    <path d="M4 5h16l1 4H3l1-4Z" />
    <path d="M4 9v9.5A2.5 2.5 0 0 0 6.5 21h11a2.5 2.5 0 0 0 2.5-2.5V9M9 13h6" />
  </svg>
)
export const IconPin = (p: P) => (
  <svg {...base(p)}>
    <path d="M9 3.5h6l-.8 5.2 3.3 3.1H14l-2 8.7-2-8.7H6.5l3.3-3.1L9 3.5Z" />
  </svg>
)
export const IconSettings = (p: P) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z" />
  </svg>
)
export const IconPanelBottom = (p: P) => (
  <svg {...base(p)}>
    <rect x="3" y="4" width="18" height="16" rx="2.5" />
    <path d="M3 14.5h18" />
  </svg>
)
export const IconPanelRight = (p: P) => (
  <svg {...base(p)}>
    <rect x="3" y="4" width="18" height="16" rx="2.5" />
    <path d="M14.5 4v16" />
  </svg>
)
export const IconPanelLeft = (p: P) => (
  <svg {...base(p)}>
    <rect x="3" y="4" width="18" height="16" rx="2.5" />
    <path d="M9.5 4v16" />
  </svg>
)
export const IconCloud = (p: P) => (
  <svg {...base(p)}>
    <path d="M7 18.5a4.5 4.5 0 0 1-.6-9 6 6 0 0 1 11.5 1.6A3.8 3.8 0 0 1 17.5 18.5H7Z" />
  </svg>
)
export const IconLaptop = (p: P) => (
  <svg {...base(p)}>
    <rect x="4.5" y="5" width="15" height="10" rx="1.5" />
    <path d="M2.5 19h19" />
  </svg>
)
export const IconHistory = (p: P) => (
  <svg {...base(p)}>
    <path d="M3.5 12a8.5 8.5 0 1 0 2.5-6" />
    <path d="M3.5 4.5V9H8M12 8v4.2l2.8 1.6" />
  </svg>
)
export const IconMore = (p: P) => (
  <svg {...base(p)}>
    <circle cx="5.5" cy="12" r="1.2" fill="currentColor" />
    <circle cx="12" cy="12" r="1.2" fill="currentColor" />
    <circle cx="18.5" cy="12" r="1.2" fill="currentColor" />
  </svg>
)

/* ------------------------- brand marks for agents ------------------------ */
type MarkProps = { size?: number; className?: string }

/** The Aro wordmark glyph — emerald geometric A on near-black. */
export function LogoMark({ size = 22, className }: MarkProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" className={className} role="img" aria-label="Aro mark">
      <defs>
        <linearGradient id="aro-mark-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#6EE7B7" />
          <stop offset="55%" stopColor="#10B981" />
          <stop offset="100%" stopColor="#0D9488" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="8" fill="#101214" />
      <rect x="1" y="1" width="30" height="30" rx="8" fill="none" stroke="#27272a" strokeWidth="1" />
      {/* geometric A — sharp apex, two legs, high crossbar */}
      <path d="M16 7.2 L24 25" stroke="url(#aro-mark-g)" strokeWidth="2.6" strokeLinecap="round" />
      <path d="M16 7.2 L8 25" stroke="url(#aro-mark-g)" strokeWidth="2.6" strokeLinecap="round" />
      <path d="M11.6 19.6 H20.4" stroke="url(#aro-mark-g)" strokeWidth="2.2" strokeLinecap="round" opacity="0.85" />
    </svg>
  )
}

/** Deterministic two-tone agent avatar mark. */
export function AgentMark({ glyph, from, to, size = 18, className }: MarkProps & { glyph: string; from: string; to: string }) {
  const id = `am-${glyph}-${from.replace('#', '')}`
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor={from} />
          <stop offset="100%" stopColor={to} />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="22" height="22" rx="6.5" fill={`url(#${id})`} opacity="0.18" />
      <rect x="1" y="1" width="22" height="22" rx="6.5" stroke={`url(#${id})`} strokeWidth="1.1" fill="none" opacity="0.75" />
      <text x="12" y="16.4" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="10.5" fontWeight="700" fill={`url(#${id})`}>
        {glyph}
      </text>
    </svg>
  )
}

/* ------------------------------- ide marks ------------------------------- */
export function IdeMark({ kind, size = 16, className }: MarkProps & { kind: string }) {
  const map: Record<string, { bg: string; label: string }> = {
    vscode: { bg: '#2ea3ef', label: 'VS' },
    cursor: { bg: '#9b8bff', label: 'CU' },
    windsurf: { bg: '#5ac8a0', label: 'WS' },
    zed: { bg: '#f2b33d', label: 'ZD' },
    jetbrains: { bg: '#ff6b7a', label: 'JB' },
    neovim: { bg: '#46d68f', label: 'NV' },
    emacs: { bg: '#a78bfa', label: 'EM' },
    xcode: { bg: '#5ba8ff', label: 'XC' },
  }
  const m = map[kind] ?? { bg: '#6e7890', label: '??' }
  return (
    <span
      className={cn('inline-flex shrink-0 items-center justify-center rounded-[4px] font-aro-mono font-bold', className)}
      style={{
        width: size,
        height: size,
        fontSize: size * 0.4,
        color: m.bg,
        background: `color-mix(in srgb, ${m.bg} 16%, transparent)`,
        boxShadow: `inset 0 0 0 1px color-mix(in srgb, ${m.bg} 40%, transparent)`,
      }}
    >
      {m.label}
    </span>
  )
}
