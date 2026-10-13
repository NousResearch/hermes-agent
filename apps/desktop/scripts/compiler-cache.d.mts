export function withCompilerCache<P extends { transform: { handler: (...args: any[]) => unknown } }>(
  plugin: P,
  options: {
    command: string
    cacheRoot: string
    base: string
    toolchain: string[]
    env?: Record<string, string | undefined>
  },
): P
