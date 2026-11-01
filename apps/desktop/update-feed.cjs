'use strict'

// Historical native URLs are a compatibility layout, not a channel registry.
/** @param {string} channel @param {boolean} light */
function darwinFeed(channel, light = false) {
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(channel) || channel.length > 32 ||
      /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/.test(channel)) {
    throw new TypeError('Invalid channel name')
  }
  return {
    directory: `releases/darwin/${light ? 'light/' : ''}${channel}`,
    channel,
    fileName: `${channel}-mac.yml`,
    allowPrerelease: channel !== 'stable'
  }
}

// The feed origin is baked into app-update.yml and trusted by every
// installed client for the life of the build, so the CI var must be a
// canonical https base: no credentials, query, or fragment, spelled exactly
// as the URL parser re-serializes it (a trailing slash is tolerated, as the
// Python release scripts do). Anything else is a misconfigured release
// environment and must fail the build, not ship.
/** @param {string | undefined} raw @returns {string | undefined} */
function feedBaseUrl(raw) {
  if (raw === undefined || raw === '') return undefined
  let url
  try {
    url = new URL(raw)
  } catch {
    throw new TypeError(`CLOUDFLARE_R2_PUBLIC_URL is not a URL: ${raw}`)
  }
  const canonical = url.origin + url.pathname.replace(/\/+$/, '')
  if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash || canonical !== raw.replace(/\/+$/, '')) {
    throw new TypeError(`CLOUDFLARE_R2_PUBLIC_URL must be a canonical https base (got ${raw}, expected ${canonical})`)
  }
  return canonical
}

module.exports = { darwinFeed, feedBaseUrl }
