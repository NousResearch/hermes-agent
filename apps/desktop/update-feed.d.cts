interface DarwinFeed {
  /** Feed directory key under the public bucket, no trailing slash.
   *  e.g. "releases/darwin/stable" | "releases/darwin/light/canary" */
  directory: string
  /** Normalized channel. "stable" | "canary" */
  channel: 'stable' | 'canary'
  /** electron-updater manifest filename. e.g. "stable-mac.yml" */
  fileName: string
  /** True only for the canary channel. */
  allowPrerelease: boolean
}

/**
 * Feed layout contract shared by the desktop runtime and the release
 * pipeline. `light` selects the Light-variant feed directory.
 * The generic-provider feed URL is PUBLIC_URL + '/' + directory + '/' + fileName.
 */
declare function darwinFeed(channel: 'stable' | 'canary', light?: boolean): DarwinFeed

export = { darwinFeed }
