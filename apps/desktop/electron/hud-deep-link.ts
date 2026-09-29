export function isHudSummonDeepLink(url: unknown): boolean {
  return typeof url === 'string' && url.startsWith('hermes://hud/summon')
}
