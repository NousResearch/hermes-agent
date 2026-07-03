import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Vitest runs the frontend's JS-level unit/behaviour tests (pure logic + the
// accessible dialog hook via jsdom). The Python static guardrails remain the
// coarse tripwire; these give executable coverage of runtime behaviour.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
