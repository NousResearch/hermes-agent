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
import type { ComponentPropsWithoutRef, ComponentType, SVGProps } from 'react'
import { forwardRef, memo } from 'react'

import { AvatarChip } from '@/components/ui/avatar-chip'
import { Globe, Link as LinkIcon, MessageSquareText } from '@/lib/icons'

// ---------------------------------------------------------------------------
// Photon brand icon — three diagonal rounded bars (the Photon logo mark).
// Rendered at ~14 px inside the PlatformAvatar so the bars are kept thick
// enough to stay legible. At small sizes the bars blend into a distinctive
// silhouette; the wide triangular spacing preserves the logo's identity.
// ---------------------------------------------------------------------------
function PhotonIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="currentColor" viewBox="0 0 24 24" {...props}>
      <rect height="10" rx="1.25" transform="rotate(15 14 7.5)" width="2.5" x="12.75" y="2.5" />
      <rect height="10" rx="1.25" transform="rotate(15 8 13)" width="2.5" x="6.75" y="8" />
      <rect height="10" rx="1.25" transform="rotate(15 16 18)" width="2.5" x="14.75" y="13" />
    </svg>
  )
}

// Official A2A Protocol color icon (a2aproject/A2A docs/assets/a2a_logo/icon/color).
// Brand fill is #2874d7. Letter cutouts stay transparent so the GUI shows through.
function A2AIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 199.067 198.437" {...props}>
      <g fill="#2874d7">
        <path d="M181.924,127.367c3.138-8.975,4.856-18.61,4.856-28.642,0-47.977-39.032-87.009-87.009-87.009-10.102,0-19.804,1.737-28.832,4.916,1.652,2.336,3.018,4.812,4.092,7.382,7.783-2.584,16.1-3.987,24.74-3.987,43.394,0,78.698,35.304,78.698,78.698,0,8.637-1.405,16.95-3.988,24.732-4.417-1.745-9.225-2.714-14.262-2.714-21.455,0-38.847,17.393-38.847,38.847s17.393,38.847,38.847,38.847,38.847-17.393,38.847-38.847c0-13.416-6.801-25.243-17.143-32.223ZM183.236,184.307c-2.133.836-3.619.216-4.455-1.863l-1.936-4.697-2.438-5.914h-25.232l-2.325,5.914-1.846,4.697c-.837,2.106-2.296,2.727-4.375,1.863-2.16-.836-2.808-2.322-1.944-4.455l.855-2.105,4.991-12.29,13.431-33.07c.81-1.782,2.079-2.673,3.807-2.673h.162c1.755.081,2.942.972,3.564,2.673l18.493,44.78.24.58.869,2.105c.864,2.133.243,3.619-1.863,4.455Z" />
        <polygon points="46.001 43.115 48.811 43.115 38.726 18.572 29.046 43.115 32.747 43.115 46.001 43.115" />
        <path d="M99.771,177.422c-43.394,0-78.698-35.304-78.698-78.698,0-8.252,1.285-16.208,3.651-23.689,4.379,1.71,9.139,2.659,14.123,2.659,21.455,0,38.847-17.393,38.847-38.847S60.302,0,38.847,0,0,17.393,0,38.847c0,13.462,6.849,25.321,17.25,32.292-2.907,8.673-4.489,17.947-4.489,27.585,0,47.977,39.032,87.009,87.009,87.009,9.809,0,19.243-1.633,28.047-4.639-1.581-2.372-2.866-4.882-3.868-7.477-7.622,2.466-15.747,3.805-24.179,3.805ZM15.884,57.938L35.162,10.472c.81-1.782,2.079-2.673,3.807-2.673h.162c1.755.081,2.942.972,3.564,2.673l18.319,44.36,1.111,2.691.171.414c.864,2.133.243,3.619-1.863,4.455-2.133.836-3.619.216-4.455-1.863l-1.239-3.006-3.135-7.605h-25.232l-2.989,7.605-1.182,3.006c-.837,2.106-2.296,2.727-4.375,1.863-2.16-.836-2.808-2.322-1.944-4.455Z" />
        <path d="M72.381,126.727c15.113,15.113,39.616,15.113,54.729,0,15.113-15.113,15.113-39.616,0-54.729-15.113-15.113-39.616-15.113-54.729,0-15.113,15.113-15.113,39.616,0,54.729ZM85.081,73.609c2.43-1.876,5.873-2.815,10.328-2.815h8.666c5.076,0,8.842,1.229,11.3,3.685.657.657,1.219,1.415,1.701,2.26,1.318,2.313,1.985,5.321,1.985,9.039,0,3.241-.797,5.947-2.39,8.12-1.593,2.174-3.685,4.091-6.277,5.751-1.026.657-2.105,1.316-3.19,1.976-1.657,1.006-3.359,2.015-5.153,3.026-2.16,1.242-4.206,2.484-6.136,3.726-1.679,1.081-3.131,2.324-4.38,3.71-.187.208-.383.409-.56.624-1.364,1.648-2.262,3.713-2.693,6.197h26.73c1.428,0,2.405.452,2.934,1.352.311.529.468,1.211.468,2.05,0,2.269-1.134,3.402-3.402,3.402h-30.294c-2.322,0-3.484-1.134-3.484-3.402,0-.709.04-1.381.084-2.05.181-2.74.714-5.153,1.637-7.204,1.024-2.278,2.321-4.225,3.862-5.879.185-.199.359-.411.552-.601,1.796-1.768,3.706-3.266,5.731-4.495.262-.159.5-.299.757-.455,1.729-1.044,3.367-2.017,4.872-2.887,2.997-1.701,5.501-3.172,7.513-4.415.372-.23.709-.462,1.047-.694,1.486-1.02,2.654-2.057,3.468-3.113.999-1.296,1.499-2.876,1.499-4.739,0-.884-.048-1.681-.135-2.406-.229-1.912-.755-3.281-1.587-4.094-1.148-1.12-3.3-1.681-6.46-1.681h-8.666c-2.539,0-4.415.338-5.63,1.012-1.215.675-1.985,1.931-2.309,3.767-.046.264-.103.511-.17.744-.208.722-.515,1.298-.923,1.727-.54.567-1.363.851-2.47.851-1.134,0-1.998-.304-2.592-.911-.411-.42-.642-.982-.717-1.666-.034-.306-.046-.628-.012-.987.437-2.932,1.432-5.286,2.975-7.073.46-.533.963-1.021,1.521-1.453Z" />
        <polygon points="151.849 165.029 153.78 165.029 168.759 165.029 171.613 165.029 161.528 140.487 151.849 165.029" />
      </g>
    </svg>
  )
}

// Signal's Simple Icons glyph uses currentColor. AvatarChip tints from `color`,
// so a transparent tile would also wipe the bubble unless fill is locked here.
function SignalIcon(props: SVGProps<SVGSVGElement>) {
  return <SiSignal {...props} fill="#3A76F0" />
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
  signal: { Icon: SignalIcon, color: 'transparent', kind: 'brand' },
  whatsapp: { Icon: SiWhatsapp, color: '#25D366', kind: 'brand' },
  bluebubbles: { Icon: SiApple, color: '#0BD318', kind: 'brand' },
  photon: { Icon: PhotonIcon, color: '#6366F1', kind: 'brand' },
  homeassistant: { Icon: SiHomeassistant, color: '#18BCF2', kind: 'brand' },
  email: { Icon: SiGmail, color: '#EA4335', kind: 'brand' },
  sms: { Icon: MessageSquareText, color: '#F43F5E', kind: 'generic' },
  webhook: { Icon: LinkIcon, color: '#71717A', kind: 'generic' },
  api_server: { Icon: Globe, color: '#64748B', kind: 'generic' },
  weixin: { Icon: SiWechat, color: '#07C160', kind: 'brand' },
  qqbot: { Icon: SiQq, color: '#EB1923', kind: 'brand' },
  yuanbao: { Icon: SiBilibili, color: '#FB7299', kind: 'brand' },
  a2a: { Icon: A2AIcon, color: 'transparent', kind: 'brand' }
}

interface PlatformAvatarProps extends Omit<ComponentPropsWithoutRef<'span'>, 'children'> {
  platformId: string
  platformName: string
}

// forwardRef + spreading ...rest is required so a wrapping <Tip> (Radix
// Tooltip's `asChild`) can actually attach its trigger: asChild clones this
// component and injects a ref plus pointer/focus/aria handlers onto it. A
// plain function component with no ref/rest forwarding drops all of that
// silently — the tooltip renders but never opens (#67500).
export const PlatformAvatar = memo(
  forwardRef<HTMLSpanElement, PlatformAvatarProps>(function PlatformAvatar(
    { className, platformId, platformName, ...rest },
    ref
  ) {
    return (
      <AvatarChip
        aria-hidden="true"
        brand={PLATFORM_ICONS[platformId]}
        className={className}
        name={platformName}
        ref={ref}
        {...rest}
      />
    )
  })
)
