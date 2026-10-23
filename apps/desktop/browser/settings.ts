import { selectServerDirectory } from './directory-picker'
import { connection, requestJson } from './transport'

function preferenceScope() {
  const profile = connection().profile || 'default'

  return { profile, key: `hermes-browser-default-cwd:${window.__HERMES_BASE_PATH__ || ''}:${profile}` }
}

/** Browser preferences are scoped to this backend URL and profile, without secrets. */
export const settings: Window['hermesDesktop']['settings'] = {
  async getDefaultProjectDir() {
    const { profile, key } = preferenceScope()
    const { cwd } = await requestJson<{ cwd: string }>({ path: '/api/fs/default-cwd', profile })
    const dir = localStorage.getItem(key) || null

    return { dir, defaultLabel: cwd, resolvedCwd: dir || cwd }
  },
  async pickDefaultProjectDir() {
    const [dir] = await selectServerDirectory(preferenceScope().profile)

    return { canceled: !dir, dir: dir || null }
  },
  async setDefaultProjectDir(dir) {
    const { key, profile } = preferenceScope()

    if (dir) {
      const result = await requestJson<{ error?: string }>({ path: `/api/fs/list?path=${encodeURIComponent(dir)}`, profile })

      if (result.error) {throw new Error(result.error)}
      localStorage.setItem(key, dir)
    } else {
      localStorage.removeItem(key)
    }

    return { dir }
  }
}
