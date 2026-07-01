import type { MessagingPlatformInfo } from '@/types/hermes'

// ApexNodes is a China-first managed product, so the 消息平台 (messaging) picker
// hides platforms that aren't usable from mainland China (blocked / need a VPN).
//
// Unlike the model picker (see lib/provider-allowlist.ts, an ALLOWLIST because
// foreign providers are open-ended — 200+ via aggregators), the messaging
// platform set is small and closed, and the *domestic* set is what keeps growing
// (钉钉 / 飞书 / 企业微信 / 个人微信 / QQ / 元宝 + future ones). So here we exclude
// by an explicit FOREIGN denylist — that guarantees we never accidentally hide a
// domestic platform we didn't enumerate. Email / webhook / API server are
// region-neutral and stay.
//
// Display-only: the runtime platform adapters (gateway/platforms/*.py) are
// untouched — a hidden platform is simply not offered in the desktop UI.
//
// To hide a newly-added foreign platform: drop its runtime id into FOREIGN_PLATFORM_IDS.
export const FOREIGN_PLATFORM_IDS: ReadonlySet<string> = new Set([
  'telegram',
  'discord',
  'slack',
  'mattermost',
  'matrix',
  'signal',
  'whatsapp',
  'bluebubbles', // iMessage bridge — needs a Mac + Apple ID, impractical in CN
  'homeassistant', // smart-home hub, not a CN messaging app
  'sms' // Twilio SMS gateway — no mainland delivery
])

/** True when a messaging platform should show in the China-first picker (i.e. it
 *  is not on the foreign denylist). Matched case-insensitively on the runtime id. */
export function isDomesticPlatform(id: string): boolean {
  return !FOREIGN_PLATFORM_IDS.has(String(id || '').trim().toLowerCase())
}

/** Keep only the messaging platforms the China-first picker should show. Order
 *  is preserved; runtime adapters for hidden platforms remain fully intact. */
export function filterDomesticPlatforms(
  platforms: MessagingPlatformInfo[]
): MessagingPlatformInfo[] {
  return platforms.filter(platform => isDomesticPlatform(platform.id))
}
