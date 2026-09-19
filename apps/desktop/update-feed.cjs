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

module.exports = { darwinFeed }
