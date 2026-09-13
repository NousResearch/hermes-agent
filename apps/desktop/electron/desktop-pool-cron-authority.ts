/** Environment markers that distinguish the primary Desktop backend from pooled helpers. */
export const poolBackendAuthorityEnv = {
  HERMES_DESKTOP: '1',
  HERMES_DESKTOP_POOL: '1'
} as const

export function withPoolBackendAuthorityEnv(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  return { ...env, ...poolBackendAuthorityEnv }
}
