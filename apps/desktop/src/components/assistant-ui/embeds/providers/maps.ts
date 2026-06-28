import { bareHost, type EmbedMatcher, type FrameEmbed } from './types'

// `@lat,lng` (optionally `,<zoom>z`) as it appears in Google Maps URLs.
const LATLNG_RE = /@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)(?:,(\d+(?:\.\d+)?)z)?/

function googleMapsEmbed(url: URL): FrameEmbed | null {
  const host = bareHost(url.hostname)

  if (host !== 'google.com' && host !== 'maps.google.com' && !host.startsWith('google.')) {
    return null
  }

  const isMapsPath = host.startsWith('maps.') || url.pathname.startsWith('/maps')

  if (!isMapsPath) {
    return null
  }

  // Prefer explicit coordinates; then a `q=` query; then a `/place/<name>`.
  const coords = url.pathname.match(LATLNG_RE)
  const placeName = url.pathname.match(/\/place\/([^/@]+)/)
  const query = url.searchParams.get('q') || url.searchParams.get('query')
  let q = ''
  let zoom = ''

  if (coords) {
    q = `${coords[1]},${coords[2]}`
    zoom = coords[3] ? String(Math.round(Number(coords[3]))) : ''
  } else if (query) {
    q = query
  } else if (placeName) {
    q = decodeURIComponent(placeName[1].replace(/\+/g, ' '))
  }

  if (!q) {
    return null
  }

  // `output=embed` is the long-standing keyless Maps embed surface.
  const params = new URLSearchParams({ output: 'embed', q })

  if (zoom) {
    params.set('z', zoom)
  }

  return {
    aspectRatio: 16 / 10,
    embedUrl: `https://maps.google.com/maps?${params.toString()}`,
    id: `googlemaps:${q}${zoom ? `@${zoom}` : ''}`,
    label: 'Google Maps',
    maxWidth: 640,
    provider: 'googlemaps',
    renderer: 'frame',
    sourceUrl: url.toString()
  }
}

function openStreetMapEmbed(url: URL): FrameEmbed | null {
  if (bareHost(url.hostname) !== 'openstreetmap.org') {
    return null
  }

  // State lives in the fragment: `#map=<zoom>/<lat>/<lng>`.
  const match = url.hash.match(/map=(\d+(?:\.\d+)?)\/(-?\d+(?:\.\d+)?)\/(-?\d+(?:\.\d+)?)/)

  if (!match) {
    return null
  }

  const zoom = Number(match[1])
  const lat = Number(match[2])
  const lng = Number(match[3])
  // Degrees spanned at this zoom; halved for the bbox half-extent.
  const lonDelta = 360 / 2 ** zoom
  const latDelta = lonDelta / 2

  const bbox = [lng - lonDelta / 2, lat - latDelta / 2, lng + lonDelta / 2, lat + latDelta / 2]
    .map(value => value.toFixed(5))
    .join(',')

  const params = new URLSearchParams({ bbox, layer: 'mapnik', marker: `${lat},${lng}` })

  return {
    aspectRatio: 16 / 10,
    embedUrl: `https://www.openstreetmap.org/export/embed.html?${params.toString()}`,
    id: `openstreetmap:${lat},${lng}@${zoom}`,
    label: 'OpenStreetMap',
    maxWidth: 640,
    provider: 'openstreetmap',
    renderer: 'frame',
    sourceUrl: url.toString()
  }
}

export const maps: EmbedMatcher = url => googleMapsEmbed(url) || openStreetMapEmbed(url)
