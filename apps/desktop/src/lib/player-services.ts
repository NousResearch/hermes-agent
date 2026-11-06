/**
 * Native player service catalog. Experimental providers are only entry-point and
 * navigation candidates; this metadata does not claim sign-in or playback works.
 */
export interface PlayerServiceDefinition {
  id: string
  label: string
  homeUrl: string
  requiresWidevine: boolean
  validation: 'existing' | 'experimental'
  /** Exact HTTPS hostnames. Keep redirects out until independently verified. */
  serviceHosts: readonly string[]
  /** Exact HTTPS sign-in/consent hostnames verified for this service. */
  authHosts: readonly string[]
  /** Embedded sign-in dependencies; never top-level, popup or media authority. */
  authFrameRules?: readonly { hostname: string; pathnamePrefix: string }[]
  /** Exact origins allowed to request the runtime's protected-media capability. */
  mediaKeyHosts: readonly string[]
}

export const PLAYER_SERVICES = [
  {
    id: 'youtube',
    label: 'YouTube',
    homeUrl: 'https://www.youtube.com/',
    requiresWidevine: false,
    validation: 'existing',
    serviceHosts: ['www.youtube.com', 'youtube.com', 'm.youtube.com', 'tv.youtube.com', 'music.youtube.com'],
    authHosts: ['accounts.google.com', 'accounts.youtube.com', 'consent.google.com', 'myaccount.google.com'],
    mediaKeyHosts: ['youtube.com', 'www.youtube.com', 'tv.youtube.com']
  },
  {
    id: 'tv',
    label: 'YouTube TV',
    homeUrl: 'https://tv.youtube.com/',
    requiresWidevine: true,
    validation: 'existing',
    serviceHosts: ['www.youtube.com', 'youtube.com', 'm.youtube.com', 'tv.youtube.com', 'music.youtube.com'],
    authHosts: ['accounts.google.com', 'accounts.youtube.com', 'consent.google.com', 'myaccount.google.com'],
    mediaKeyHosts: ['youtube.com', 'www.youtube.com', 'tv.youtube.com']
  },
  {
    id: 'netflix',
    label: 'Netflix',
    homeUrl: 'https://www.netflix.com/',
    requiresWidevine: true,
    validation: 'experimental',
    serviceHosts: ['www.netflix.com'],
    authHosts: ['netflix.com'],
    mediaKeyHosts: ['www.netflix.com']
  },
  {
    id: 'prime-video',
    label: 'Prime Video',
    homeUrl: 'https://www.primevideo.com/',
    requiresWidevine: true,
    validation: 'experimental',
    serviceHosts: ['www.primevideo.com'],
    authHosts: ['www.amazon.com'],
    mediaKeyHosts: ['www.primevideo.com']
  },
  {
    id: 'disney-plus',
    label: 'Disney+',
    homeUrl: 'https://www.disneyplus.com/',
    requiresWidevine: true,
    validation: 'experimental',
    serviceHosts: ['www.disneyplus.com'],
    authHosts: [],
    mediaKeyHosts: ['www.disneyplus.com']
  },
  {
    id: 'hulu',
    label: 'Hulu',
    homeUrl: 'https://www.hulu.com/',
    requiresWidevine: true,
    validation: 'experimental',
    serviceHosts: ['www.hulu.com'],
    authHosts: ['auth.hulu.com'],
    mediaKeyHosts: ['www.hulu.com'],
    authFrameRules: [
      // Google documents these frame-src paths for reCAPTCHA. Hulu's login
      // embeds the challenge; it is not a destination the player may open.
      { hostname: 'www.google.com', pathnamePrefix: '/recaptcha/' },
      { hostname: 'recaptcha.google.com', pathnamePrefix: '/recaptcha/' }
    ]
  },
  {
    id: 'hbo-max',
    label: 'HBO Max',
    homeUrl: 'https://www.hbomax.com/',
    requiresWidevine: true,
    validation: 'experimental',
    serviceHosts: ['www.hbomax.com', 'hbomax.com', 'play.hbomax.com'],
    authHosts: [],
    mediaKeyHosts: ['www.hbomax.com', 'hbomax.com', 'play.hbomax.com']
  },
  {
    id: 'apple-tv',
    label: 'Apple TV+',
    homeUrl: 'https://tv.apple.com/',
    requiresWidevine: true,
    validation: 'experimental',
    serviceHosts: ['tv.apple.com'],
    authHosts: [],
    mediaKeyHosts: ['tv.apple.com']
  },
  {
    id: 'peacock',
    label: 'Peacock',
    homeUrl: 'https://www.peacocktv.com/',
    requiresWidevine: true,
    validation: 'experimental',
    serviceHosts: ['www.peacocktv.com'],
    authHosts: [],
    mediaKeyHosts: ['www.peacocktv.com']
  },
  {
    id: 'paramount-plus',
    label: 'Paramount+',
    homeUrl: 'https://www.paramountplus.com/',
    requiresWidevine: true,
    validation: 'experimental',
    serviceHosts: ['www.paramountplus.com'],
    authHosts: [],
    mediaKeyHosts: ['www.paramountplus.com']
  }
] as const satisfies readonly PlayerServiceDefinition[]

export type PlayerSource = (typeof PLAYER_SERVICES)[number]['id']
export type PlayerService = (typeof PLAYER_SERVICES)[number]

const servicesById = new Map<PlayerSource, PlayerService>(
  PLAYER_SERVICES.map(service => [service.id, service] as const)
)

export function getPlayerService(source: PlayerSource): PlayerService {
  return servicesById.get(source)!
}

export function isPlayerSource(value: unknown): value is PlayerSource {
  return typeof value === 'string' && servicesById.has(value as PlayerSource)
}
