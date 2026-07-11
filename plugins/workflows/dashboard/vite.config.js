import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(fileURLToPath(import.meta.url));
const outDir = process.env.WORKFLOWS_PLUGIN_OUT_DIR || path.join(root, "dist");

export default defineConfig({
  root,
  build: {
    emptyOutDir: true,
    outDir,
    cssCodeSplit: false,
    minify: false,
    lib: {
      entry: path.join(root, "src/index.js"),
      formats: ["iife"],
      name: "HermesWorkflowsPlugin",
      fileName: () => "index.js",
      cssFileName: "style",
    },
    rollupOptions: {
      output: { assetFileNames: "style.css" },
    },
  },
});