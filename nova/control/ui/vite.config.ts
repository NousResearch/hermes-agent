import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Builds straight into the directory the stdlib control server already serves, so shipping
// the dashboard needs no Python change and no second server. Assets are hashed and
// root-relative: `nova/control/server.py` resolves `/assets/x.js` under STATIC_DIR.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  build: {
    outDir: path.resolve(import.meta.dirname, "..", "static"),
    emptyOutDir: false, // app.css/app.js from the previous dashboard are removed deliberately
    rollupOptions: { output: { entryFileNames: "assets/[name]-[hash].js",
                               chunkFileNames: "assets/[name]-[hash].js",
                               assetFileNames: "assets/[name]-[hash][extname]" } },
  },
});
