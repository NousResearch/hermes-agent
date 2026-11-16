import { httpApi, setHttpToken } from './http'
import { mockApi, setMockToken } from './mock'
import type { AdminApi } from './types'

export { ApiError } from './errors'
export * from './types'

/** 默认使用示例数据（后端未就绪）；设置 VITE_ADMIN_MOCK=false 后调用真实的 /api/admin/v1 */
export const api: AdminApi = import.meta.env.VITE_ADMIN_MOCK === 'false' ? httpApi : mockApi

export function setToken(token: string): void {
  setHttpToken(token)
  setMockToken(token)
}
