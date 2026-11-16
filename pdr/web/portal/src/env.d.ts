/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 'true'（默认）：使用内置模拟接口，无需后端即可体验；'false'：调用真实 /api/portal/v1 */
  readonly VITE_PORTAL_MOCK?: string
  /** 用户端接口前缀，默认 /api/portal/v1 */
  readonly VITE_PORTAL_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<object, object, unknown>
  export default component
}
