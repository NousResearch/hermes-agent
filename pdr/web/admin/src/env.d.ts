/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 'true'（默认）：使用内置示例数据，无需后端即可体验；'false'：调用真实 /api/admin/v1 */
  readonly VITE_ADMIN_MOCK?: string
  /** 管理端接口前缀，默认 /api/admin/v1 */
  readonly VITE_ADMIN_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<object, object, unknown>
  export default component
}
