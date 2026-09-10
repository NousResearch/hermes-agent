import {
  SiDiscord,
  SiGmail,
  SiGooglechat,
  SiHomeassistant,
  SiLine,
  SiMattermost,
  SiNtfy,
  SiQq,
  SiSignal,
  SiTelegram,
  SiWechat,
  SiWhatsapp
} from '@icons-pack/react-simple-icons'
import type { ComponentPropsWithoutRef, ComponentType, SVGProps } from 'react'
import { forwardRef, memo } from 'react'

import dingtalkIconUrl from '@/assets/brand/dingtalk-icon.png'
import larkIconUrl from '@/assets/brand/lark-icon.svg?url'
import slackIconUrl from '@/assets/brand/slack-logo.svg?url'
import teamsIconUrl from '@/assets/brand/teams-icon.svg?url'
import { AvatarChip } from '@/components/ui/avatar-chip'
import { Globe, Link as LinkIcon, MessageSquareText } from '@/lib/icons'

function PhotonIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="currentColor" viewBox="0 0 24 24" {...props}>
      <rect height="10" rx="1.25" transform="rotate(15 14 7.5)" width="2.5" x="12.75" y="2.5" />
      <rect height="10" rx="1.25" transform="rotate(15 8 13)" width="2.5" x="6.75" y="8" />
      <rect height="10" rx="1.25" transform="rotate(15 16 18)" width="2.5" x="14.75" y="13" />
    </svg>
  )
}

function BlueBubblesIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="none" viewBox="0 0 24 24" {...props}>
      <g transform="translate(2 2) scale(0.014648)">
        <path
          d="m 282.22721,1212.3333 c -0.32825,-0.55 5.69752,-7.9 13.39061,-16.3334 33.18551,-36.3787 58.29388,-71.8345 75.23815,-106.2446 3.73304,-7.581 6.49275,-14.9172 6.24837,-16.6103 -0.33979,-2.3541 -4.59263,-5.3585 -19.01354,-13.432 -35.70283,-19.988 -61.53807,-37.8784 -94.09756,-65.16095 C 171.16156,916.76576 112.27757,812.10599 97.896372,699.33329 94.781572,674.90803 95.839037,616.72973 99.883646,589.99996 118.62238,466.16054 192.89128,353.15325 307.52801,274.04905 c 57.51358,-39.6868 119.86684,-68.77221 189.80529,-88.5369 92.27163,-26.07607 195.5171,-32.19794 293.28166,-17.38993 105.07046,15.91459 206.27664,57.29422 286.05164,116.95639 75.1003,56.16607 128.5993,122.56089 162.9707,202.25469 14.9183,34.58952 25.5556,73.82394 31.2695,115.33333 2.6514,19.262 2.2488,79.05098 -0.6734,99.99999 -7.1973,51.59646 -20.2747,93.78278 -43.0198,138.77775 -22.0789,43.67702 -49.2989,81.42109 -85.8471,119.03834 -86.2635,88.78679 -206.62891,149.79589 -339.60566,172.13449 -90.53541,15.2089 -180.99947,12.7908 -270.90069,-7.2411 -7.2654,-1.6189 -12.28279,-2.1026 -14.20992,-1.3699 -1.63845,0.6229 -11.60503,6.331 -22.14796,12.6847 -36.3239,21.8903 -86.62516,44.5647 -128.50229,57.925 -38.8873,12.4065 -81.55656,21.4307 -83.77277,17.7174 z"
          fill="none"
          stroke="currentColor"
          strokeLinejoin="miter"
          strokeWidth="96"
        />
        <circle cx="468.89728" cy="654.54889" fill="currentColor" r="80" />
        <circle cx="682.66663" cy="654.54889" fill="currentColor" r="80" />
        <circle cx="896.43597" cy="654.54889" fill="currentColor" r="80" />
      </g>
    </svg>
  )
}

function MatrixIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="currentColor" viewBox="0 0 24 24" {...props}>
      <g transform="translate(2 2) scale(0.038462)">
        <path d="M13.7,11.9v496.2h35.7V520H0V0h49.4v11.9H13.7z" fill="currentColor" />
        <path
          d="M166.3,169.2v25.1h0.7c6.7-9.6,14.8-17,24.2-22.2c9.4-5.3,20.3-7.9,32.5-7.9c11.7,0,22.4,2.3,32.1,6.8  c9.7,4.5,17,12.6,22.1,24c5.5-8.1,13-15.3,22.4-21.5c9.4-6.2,20.6-9.3,33.5-9.3c9.8,0,18.9,1.2,27.3,3.6c8.4,2.4,15.5,6.2,21.5,11.5  c6,5.3,10.6,12.1,14,20.6c3.3,8.5,5,18.7,5,30.7v124.1h-50.9V249.6c0-6.2-0.2-12.1-0.7-17.6c-0.5-5.5-1.8-10.3-3.9-14.3  c-2.2-4.1-5.3-7.3-9.5-9.7c-4.2-2.4-9.9-3.6-17-3.6c-7.2,0-13,1.4-17.4,4.1c-4.4,2.8-7.9,6.3-10.4,10.8c-2.5,4.4-4.2,9.4-5,15.1  c-0.8,5.6-1.3,11.3-1.3,17v103.3h-50.9v-104c0-5.5-0.1-10.9-0.4-16.3c-0.2-5.4-1.3-10.3-3.1-14.9c-1.8-4.5-4.8-8.2-9-10.9  c-4.2-2.7-10.3-4.1-18.5-4.1c-2.4,0-5.6,0.5-9.5,1.6c-3.9,1.1-7.8,3.1-11.5,6.1c-3.7,3-6.9,7.3-9.5,12.9c-2.6,5.6-3.9,13-3.9,22.1  v107.6h-50.9V169.2H166.3z"
          fill="currentColor"
        />
        <path d="M506.3,508.1V11.9h-35.7V0H520v520h-49.4v-11.9H506.3z" fill="currentColor" />
      </g>
    </svg>
  )
}

function WeComIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="currentColor" viewBox="0 0 24 24" {...props}>
      {/* Optical-only adjustment: keep the official paths intact while using the
          24px chip more evenly. */}
      <g transform="translate(0.9 0.9) scale(0.46)">
        <path
          d="M36.75,37.19s-.09-.02-.14-.03c-.05,0-.09-.01-.14-.02-1.37-.26-2.68-.91-3.74-1.97-.08-.08-.15-.15-.22-.23h0c-.22-.22-.57-.22-.78,0-.2,.2-.22,.52-.04,.74,.01,.01,.02,.03,.04,.04,.02,.02,.05,.04,.08,.06,.05,.05,.11,.1,.16,.15,1.04,1.04,1.69,2.31,1.95,3.65,0,.08,.01,.15,.02,.22,.01,.08,.03,.16,.05,.24,.1,.38,.3,.73,.6,1.03,.9,.9,2.37,.9,3.27,0,.9-.9,.9-2.37,0-3.27-.32-.32-.71-.52-1.12-.62"
          fill="currentColor"
          fillRule="evenodd"
        />
        <path
          d="M44.41,31.28c-.9-.9-2.37-.9-3.27,0-.32,.32-.52,.71-.62,1.12-.01,.05-.02,.09-.03,.14,0,.05-.01,.09-.02,.14-.26,1.37-.91,2.68-1.97,3.74-.08,.08-.15,.15-.23,.22h0c-.22,.22-.22,.57,0,.78,.2,.2,.52,.22,.74,.04,.01-.01,.03-.02,.04-.04,.02-.02,.04-.05,.06-.08,.05-.05,.1-.11,.15-.16,1.04-1.04,2.31-1.69,3.65-1.95,.08,0,.15-.01,.22-.02,.08-.01,.16-.03,.24-.05,.38-.1,.73-.3,1.03-.6,.9-.9,.9-2.37,0-3.27"
          fill="currentColor"
          fillRule="evenodd"
        />
        <path
          d="M34.61,24.74c-.9,.9-.9,2.37,0,3.27,.32,.32,.71,.52,1.12,.62,.05,.01,.09,.02,.14,.03,.05,0,.09,.01,.14,.02,1.37,.26,2.68,.91,3.74,1.97,.08,.08,.15,.15,.22,.23,.22,.22,.57,.22,.78,0,.2-.2,.21-.52,.04-.74-.01-.01-.02-.03-.04-.04-.02-.02-.05-.04-.08-.06-.05-.05-.11-.1-.16-.15-1.04-1.04-1.69-2.31-1.95-3.65,0-.08-.01-.15-.02-.22-.01-.08-.03-.16-.05-.24-.1-.38-.3-.73-.6-1.03-.9-.9-2.37-.9-3.27,0"
          fill="currentColor"
          fillRule="evenodd"
        />
        <path
          d="M31.98,33.29s.01-.09,.02-.14c.26-1.37,.91-2.68,1.97-3.74,.08-.08,.15-.15,.23-.22h0c.22-.22,.22-.57,0-.78-.2-.2-.52-.22-.74-.04-.01,.01-.03,.02-.04,.04-.02,.02-.05,.05-.06,.08-.05,.05-.1,.11-.15,.16-1.04,1.04-2.31,1.69-3.65,1.95-.07,0-.15,.01-.22,.02-.08,.01-.16,.03-.24,.05-.38,.1-.73,.3-1.03,.6-.9,.9-.9,2.37,0,3.27s2.37,.9,3.27,0c.32-.32,.52-.71,.62-1.12,.01-.05,.02-.09,.03-.14"
          fill="currentColor"
          fillRule="evenodd"
        />
        <path
          d="M36.91,17.05c-.64-1.32-1.51-2.55-2.57-3.65-2.69-2.78-6.44-4.57-10.58-5.04-.74-.08-1.48-.13-2.19-.13s-1.38,.04-2.1,.12c-4.16,.45-7.94,2.23-10.64,5.01-1.07,1.1-1.94,2.32-2.59,3.64-.88,1.79-1.33,3.69-1.33,5.65,0,2.52,.77,5.01,2.22,7.19,.74,1.11,1.94,2.5,3.04,3.48h0s-.5,3.93-.5,3.93c-.02,.05-.04,.11-.05,.16-.01,.05-.01,.1-.02,.16,0,.04-.01,.08-.01,.12,0,.04,0,.09,.01,.13,.07,.65,.6,1.15,1.27,1.15,.23,0,.45-.07,.63-.17,0,0,.01,0,.02,0,.03-.02,.06-.03,.08-.05l1.19-.6,3.56-1.79c1.02,.29,2.04,.48,3.11,.6,.7,.08,1.4,.12,2.1,.12s1.45-.04,2.19-.13c1.46-.17,2.86-.5,4.2-.98-.14-.05-.29-.11-.42-.19-.82-.47-1.24-1.35-1.15-2.23-.95,.3-1.95,.52-2.97,.64-.62,.07-1.24,.11-1.83,.11s-1.17-.03-1.76-.1c-.12-.01-.24-.03-.36-.05-.8-.11-1.59-.27-2.35-.49-.16-.05-.32-.07-.49-.07-.26,0-.52,.07-.78,.2-.03,.02-.07,.03-.1,.05l-2.92,1.72-.13,.07h0c-.06,.04-.1,.05-.13,.05-.11,0-.19-.09-.19-.2l.11-.46c.03-.12,.08-.29,.13-.5,.06-.24,.14-.53,.21-.81,.08-.32,.17-.64,.24-.91,.03-.11,.06-.23,.06-.37,0-.39-.19-.76-.5-.99-.16-.12-.32-.24-.49-.38-.26-.21-.5-.43-.74-.65-.66-.63-1.24-1.31-1.73-2.05-1.15-1.73-1.76-3.69-1.76-5.67,0-1.54,.35-3.03,1.05-4.45,.52-1.05,1.22-2.04,2.07-2.92,2.22-2.28,5.35-3.75,8.8-4.12,.6-.07,1.19-.1,1.76-.1,.59,0,1.21,.04,1.83,.11,3.44,.39,6.55,1.86,8.75,4.14,.85,.88,1.55,1.87,2.06,2.93,.68,1.4,1.03,2.89,1.03,4.41,0,.16,0,.32-.02,.47,.89-.55,2.07-.44,2.84,.33,.04,.04,.07,.08,.11,.12,.03-.32,.04-.65,.04-.98,0-1.94-.44-3.83-1.3-5.61"
          fill="currentColor"
          fillRule="evenodd"
        />
      </g>
    </svg>
  )
}

// Preserve Raft's official icon-only geometry and two-tone treatment from
// raft.build. The adjacent RAFT wordmark is intentionally omitted at 24px.
function RaftIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="none" viewBox="0 0 113 104" {...props}>
      <path
        d="M112.754 38.7427L108.8 16.3029C108.228 13.0949 106.323 10.5063 103.797 8.4577C102.543 7.44132 98.0803 3.58224 96.2699 2.18472C93.7131 0.183714 90.3622 -0.324477 87.186 0.183714L21.8199 11.8404C16.8809 12.5709 12.9424 13.4761 9.90915 17.5893C6.39945 22.3377 7.43172 26.9273 8.25753 31.6122L11.3384 49.1289C11.4655 49.8435 11.6719 50.5105 11.8943 51.1617C7.57465 51.9239 4.16024 53.4803 2.07983 56.5453C-0.175272 59.8644 -0.349963 62.2942 0.396443 66.5503L3.92202 86.5603C4.82724 90.7688 8.03519 92.9286 12.7042 97.1212C18.1196 101.949 20.2635 104.649 25.679 103.696L96.3017 91.2452C102.892 90.07 107.307 83.7812 106.148 77.1906L102.273 55.1795C102.146 54.4331 101.939 53.7185 101.685 53.0356L102.972 52.7974C109.515 51.6063 113.898 45.3016 112.754 38.7586V38.7427Z"
        fill="#141111"
      />
      <path
        d="M18.3261 87.1956L85.1691 75.4119C87.3607 75.0307 88.8217 72.9345 88.4247 70.7588L85.2168 52.5274C84.8356 50.3358 82.7393 48.8748 80.5636 49.2718C78.3721 49.6529 75.9105 50.0976 74.6718 50.3199C72.7184 50.6693 69.3993 49.4624 68.8276 46.2544L67.7953 40.4102L38.8761 71.9181C37.9708 72.7439 35.9063 73.8397 33.7624 73.3315C31.9043 72.8868 30.5544 71.267 30.2209 69.393L28.2993 58.4828L13.7206 61.0555C11.529 61.4525 10.068 63.5329 10.4491 65.7086L13.6571 83.9241C14.0382 86.1157 16.1345 87.5767 18.3102 87.1797L18.3261 87.1956Z"
        fill="#FFFAEF"
      />
      <path
        d="M20.9147 45.508C21.3117 47.6996 23.3921 49.1607 25.5678 48.7795L31.4914 47.7314C34.2229 47.2549 36.8274 49.0812 37.3038 51.7969L38.3361 57.6411L67.2554 26.1332C68.5417 24.7357 70.5268 24.1799 72.369 24.7198C74.1954 25.2598 75.577 26.7844 75.9105 28.6583L77.8321 39.5368L91.7756 36.9958C93.9513 36.5988 95.3965 34.5184 95.0153 32.3427L91.7279 13.6825C91.3468 11.491 89.2505 10.0299 87.0589 10.4269L20.867 22.1948C18.6754 22.5759 17.2303 24.6722 17.6114 26.8479L20.8988 45.4921L20.9147 45.508Z"
        fill="#FFFAEF"
      />
    </svg>
  )
}

function IrcIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="none" viewBox="0 0 24 24" {...props}>
      <path d="M5 7.25h14v8.5H9.25L5 19v-3.25H5V7.25Z" stroke="currentColor" strokeLinejoin="round" strokeWidth="2" />
      <path d="M9 11.5h6" stroke="currentColor" strokeLinecap="round" strokeWidth="2" />
    </svg>
  )
}

function A2AIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="none" viewBox="0 0 24 24" {...props}>
      <circle cx="5.5" cy="12" fill="currentColor" r="2.5" />
      <circle cx="18.5" cy="12" fill="currentColor" r="2.5" />
      <path
        d="M8.5 9.25h7l-2-2M15.5 14.75h-7l2 2"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function BuzzIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="none" viewBox="0 0 24 24" {...props}>
      <path d="M8.5 8.75h7v6.5h-7z" fill="currentColor" />
      <path
        d="M6 9.75 3.5 8M6 14.25 3.5 16M18 9.75 20.5 8M18 14.25l2.5 1.75"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
      <path
        d="M10 6.25 12 4l2 2.25M10 17.75 12 20l2-2.25"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function RelayIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="none" viewBox="0 0 24 24" {...props}>
      <circle cx="5" cy="12" fill="currentColor" r="2.25" />
      <circle cx="19" cy="7" fill="currentColor" r="2.25" />
      <circle cx="19" cy="17" fill="currentColor" r="2.25" />
      <path d="m7.2 11.2 9.55-3.4M7.2 12.8l9.55 3.4" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  )
}

function SimplexChannelIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="none" viewBox="0 0 24 24" {...props}>
      <path
        d="m5 7 5 5-5 5M19 7l-5 5 5 5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2.2"
      />
      <circle cx="12" cy="12" fill="currentColor" r="1.75" />
    </svg>
  )
}

function GraphWebhookIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="none" viewBox="0 0 24 24" {...props}>
      <circle cx="6" cy="7" fill="currentColor" r="2.25" />
      <circle cx="18" cy="6" fill="currentColor" r="2.25" />
      <circle cx="17" cy="18" fill="currentColor" r="2.25" />
      <circle cx="7" cy="17" fill="currentColor" r="2.25" />
      <path
        d="m8 7 7.75-.75M17.6 8.2l-.4 7.55M14.9 17.75 9.2 17M6.5 14.75l-.25-5.5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.7"
      />
    </svg>
  )
}

function YuanbaoIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg fill="currentColor" fillRule="evenodd" viewBox="0 0 24 24" {...props}>
      <path d="M12.014.648c-6.628 0-12 5.09-12 11.367 0 6.277 5.372 11.366 12 11.366s12-5.09 12-11.366c0-6.277-5.372-11.367-12-11.367zm5.849 15.481c-4.305 3.1-10.584 2.523-13.481-1.444-2.86-3.918-1.351-9.703 2.685-13.02-1.866 1.676-2.67 5.01-1.282 6.909 1.471 2.015 4.794 1.746 6.958.113 2.435-1.84 6.036-1.794 7.234.954.91 2.208.067 4.93-2.114 6.487v.001z" />
      <path d="M14.81 14.914a.669.669 0 0 1-.536-.269l-1.02-1.37a.67.67 0 0 1 .005-.807l1.021-1.328a.669.669 0 0 1 1.06.814l-.713.926.72.964a.67.67 0 0 1-.534 1.067l-.002.003zM10.877 12.913c0 1.797-.357 2.135-1.162 2.135-.805 0-1.162-.338-1.162-2.135 0-1.798.357-2.136 1.162-2.136.805 0 1.162.338 1.162 2.136z" />
    </svg>
  )
}

type IconKind = 'brand' | 'generic'

interface PlatformIconSpec {
  Icon?: ComponentType<SVGProps<SVGSVGElement>>
  asset?: string
  color: string
  backgroundColor?: string
  glyphColor?: string
  kind: IconKind
  mask?: string
  monochrome?: boolean
  monogram?: string
}

const PLATFORM_ICONS: Record<string, PlatformIconSpec> = {
  telegram: { Icon: SiTelegram, color: '#26A5E4', kind: 'brand' },
  discord: { Icon: SiDiscord, color: '#5865F2', kind: 'brand' },
  slack: { asset: slackIconUrl, backgroundColor: '#F3EEF2', color: '#6B5870', kind: 'brand' },
  mattermost: { Icon: SiMattermost, color: '#496FA6', kind: 'brand' },
  matrix: {
    backgroundColor: '#F7F7F5',
    color: '#000000',
    kind: 'brand',
    Icon: MatrixIcon,
    monochrome: true
  },
  signal: { Icon: SiSignal, color: '#3A76F0', kind: 'brand' },
  whatsapp: { Icon: SiWhatsapp, color: '#25D366', kind: 'brand' },
  bluebubbles: { Icon: BlueBubblesIcon, color: '#5F8292', kind: 'brand' },
  photon: { Icon: PhotonIcon, color: '#6D759E', kind: 'brand' },
  homeassistant: { Icon: SiHomeassistant, color: '#4C9AB0', kind: 'brand' },
  google_chat: { Icon: SiGooglechat, color: '#5C916B', kind: 'brand' },
  irc: { Icon: IrcIcon, color: '#64748B', kind: 'generic' },
  line: { Icon: SiLine, color: '#4E9B79', kind: 'brand' },
  ntfy: { Icon: SiNtfy, color: '#5D8E84', kind: 'brand' },
  raft: { Icon: RaftIcon, color: '#D7A928', kind: 'brand' },
  simplex: { Icon: SimplexChannelIcon, color: '#668BB2', kind: 'generic' },
  teams: { color: '#74789E', kind: 'brand', mask: teamsIconUrl },
  email: { Icon: SiGmail, color: '#EA4335', kind: 'brand' },
  sms: { Icon: MessageSquareText, color: '#F43F5E', kind: 'generic' },
  webhook: { Icon: LinkIcon, color: '#71717A', kind: 'generic' },
  api_server: { Icon: Globe, color: '#64748B', kind: 'generic' },
  weixin: { Icon: SiWechat, color: '#3A9B6D', kind: 'brand' },
  wecom: { Icon: WeComIcon, color: '#5B9A63', kind: 'brand' },
  wecom_callback: { Icon: WeComIcon, color: '#5B9A63', kind: 'brand' },
  dingtalk: {
    color: '#5F89B5',
    glyphColor: '#5F89B5',
    kind: 'brand',
    mask: dingtalkIconUrl
  },
  qqbot: { Icon: SiQq, color: '#B45D66', kind: 'brand' },
  yuanbao: { Icon: YuanbaoIcon, color: '#63A886', kind: 'brand' },
  a2a: { Icon: A2AIcon, color: '#64748B', kind: 'generic' },
  buzz: { Icon: BuzzIcon, color: '#B28A4C', kind: 'generic' },
  feishu: { color: '#6689B2', kind: 'brand', mask: larkIconUrl },
  relay: { Icon: RelayIcon, color: '#64748B', kind: 'generic' },
  whatsapp_cloud: { Icon: SiWhatsapp, color: '#5A9B78', kind: 'brand' },
  msgraph_webhook: { Icon: GraphWebhookIcon, color: '#74789E', kind: 'generic' }
}

interface PlatformAvatarProps extends Omit<ComponentPropsWithoutRef<'span'>, 'children'> {
  platformId: string
  platformName: string
}

export const PlatformAvatar = memo(
  forwardRef<HTMLSpanElement, PlatformAvatarProps>(function PlatformAvatar(
    { className, platformId, platformName, ...rest },
    ref
  ) {
    const spec = PLATFORM_ICONS[platformId]

    return (
      <AvatarChip brand={spec} className={className} name={platformName} ref={ref} {...rest}>
        {spec?.asset ? (
          <img alt="" aria-hidden className="size-[58%] object-contain" data-platform-glyph="asset" src={spec.asset} />
        ) : spec?.mask ? (
          <span
            aria-hidden
            className="size-[58%]"
            data-platform-glyph="mask"
            style={{
              backgroundColor: spec.glyphColor ?? spec.color,
              maskImage: `url(${spec.mask})`,
              WebkitMaskImage: `url(${spec.mask})`,
              maskPosition: 'center',
              WebkitMaskPosition: 'center',
              maskRepeat: 'no-repeat',
              WebkitMaskRepeat: 'no-repeat',
              maskSize: 'contain',
              WebkitMaskSize: 'contain'
            }}
          />
        ) : undefined}
      </AvatarChip>
    )
  })
)
