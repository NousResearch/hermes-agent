import { httpApi } from './http'
import { mockApi } from './mock'
import type { PortalApi } from './types'

export { setHttpToken } from './http'
export * from './types'

/** 默认使用模拟接口（后端未就绪）；设置 VITE_PORTAL_MOCK=false 后调用真实的 /api/portal/v1 */
export const api: PortalApi = import.meta.env.VITE_PORTAL_MOCK === 'false' ? httpApi : mockApi
