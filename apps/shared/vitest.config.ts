import { defineConfig } from 'vitest/config'

// Shared is consumed by both the Desktop renderer and the stdio/Node clients
// (the Ink TUI), so the default environment is `node` — a suite that reaches
// for the DOM asks for it per-file with a `// @vitest-environment jsdom`
// docblock. That keeps `applyDocumentLocale`'s "is a no-op without a document"
// contract genuinely testable: a workspace-wide jsdom environment would make
// that assertion pass by accident instead of by behaviour.
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts']
  }
})
