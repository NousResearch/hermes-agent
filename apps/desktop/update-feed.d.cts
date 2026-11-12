interface NativeFeed {
  /** Feed directory key under the public bucket, no trailing slash.
   *  e.g. "releases/darwin/stable" | "releases/linux/canary" */
  directory: string
  /** Validated slug; existence is resolved through R2. */
  channel: string
  /** electron-updater manifest filename. e.g. "stable-mac.yml" | "canary-linux-arm64.yml" */
  fileName: string
  /** Legacy nonstable feeds admit prereleases. */
  allowPrerelease: boolean
}

/**
 * Feed layout contract shared by the desktop runtime and the release
 * pipeline. `light` selects the Light-variant feed directory.
 * The generic-provider feed URL is PUBLIC_URL + '/' + directory + '/' + fileName.
 */
declare function darwinFeed(channel: string, light?: boolean): NativeFeed

/**
 * The AppImage feed: "releases/linux/<channel>", with electron-updater's
 * "<channel>-linux.yml" on x64 and "<channel>-linux-<arch>.yml" elsewhere.
 */
declare function linuxFeed(channel: string, arch: string): NativeFeed

/** Validate and canonicalize the public updater feed base URL. */
declare function feedBaseUrl(raw: string | undefined): string | undefined

declare const contract: {
  darwinFeed: typeof darwinFeed
  linuxFeed: typeof linuxFeed
  feedBaseUrl: typeof feedBaseUrl
}
export = contract
