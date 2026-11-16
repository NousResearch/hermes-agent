import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 端口与后端地址由根目录 start_run.py 从 pdr/config.yaml 读出后通过环境变量传入
const port = Number(process.env.PDR_PORTAL_PORT || 5174)
const api = process.env.PDR_API_URL || 'http://127.0.0.1:8600'

export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port,
    strictPort: true,
    // 用户端只代理用户端接口；管理端接口不在这里暴露
    proxy: { '/api/portal': { target: api, changeOrigin: true } }
  },
  preview: { host: '127.0.0.1', port }
})
