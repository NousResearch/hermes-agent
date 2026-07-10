import { defineConfig, TestProjectConfiguration } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

const reactUi: TestProjectConfiguration = {
  extends: './vite.config.ts',
  test: {
    name: 'ui',
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    globals: true
  }
}

const electronNative: TestProjectConfiguration = {
  test: {
    name: 'electron',
    environment: 'node',
    include: ['electron/**/*.test.ts']
  }
}

export default defineConfig({
  test: {
    projects: [reactUi, electronNative]
  }
})
