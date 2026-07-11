import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root,
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.js"],
    passWithNoTests: true,
  },
});