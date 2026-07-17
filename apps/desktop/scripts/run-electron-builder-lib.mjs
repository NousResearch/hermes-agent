const PLATFORM_FLAGS = new Map([
  ['--linux', 'linux'],
  ['--mac', 'darwin'],
  ['--win', 'win32'],
])

export function requestedTargetPlatforms(argv = []) {
  const targets = new Set()
  for (const arg of argv) {
    const platform = PLATFORM_FLAGS.get(arg)
    if (platform) {
      targets.add(platform)
    }
  }
  return targets
}

export function shouldUseLocalElectronDist({ argv = [], hostPlatform = process.platform } = {}) {
  const targets = requestedTargetPlatforms(argv)
  if (targets.size === 0) {
    return true
  }
  for (const target of targets) {
    if (target !== hostPlatform) {
      return false
    }
  }
  return true
}

export function computeElectronBuilderArgs({
  argv = [],
  dist = null,
  hasBinary = false,
  hostPlatform = process.platform,
} = {}) {
  const args = []
  if (dist && hasBinary && shouldUseLocalElectronDist({ argv, hostPlatform })) {
    args.push(`-c.electronDist=${dist}`)
  }
  args.push(...argv)
  return args
}
