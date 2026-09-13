'use strict'

// Read the feed facts shared by the desktop runtime and Python publisher.
// update-feed.json owns the paths; neither consumer derives a second copy.
//
//   darwinFeed('stable')          → { directory: 'releases/darwin/stable',
//                                     channel: 'stable',
//                                     fileName: 'stable-mac.yml',
//                                     allowPrerelease: false }
//   darwinFeed('canary', true)    → { directory: 'releases/darwin/light/canary',
//                                     channel: 'canary',
//                                     fileName: 'canary-mac.yml',
//                                     allowPrerelease: true }
//
// A client composes its feed URL as PUBLIC_URL + '/' + feed.directory +
// '/' + feed.fileName; the producer publishes the manifest at exactly that
// key. There is no placeholder default URL — the caller supplies the base.

const feeds = require('./update-feed.json')

function darwinFeed(channel, light = false) {
  if (!Object.hasOwn(feeds, channel)) {
    throw new TypeError(`darwinFeed: unknown channel ${JSON.stringify(channel)} (expected stable|canary)`)
  }
  return { ...feeds[channel][light ? 'light' : 'bundled'] }
}

module.exports = { darwinFeed }
