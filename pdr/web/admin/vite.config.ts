import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 端口与后端地址由根目录 start_run.py 从 pdr/config.yaml 读出后通过环境变量传入
const port = Number(process.env.PDR_ADMIN_PORT || 5173)
const api = process.env.PDR_API_URL || 'http://127.0.0.1:8600'

export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port,
    strictPort: true,
    // 管理端只代理管理端接口；用户端接口不在这里暴露
    proxy: { '/api/admin': { target: api, changeOrigin: true } }
  },
  preview: { host: '127.0.0.1', port }
})
