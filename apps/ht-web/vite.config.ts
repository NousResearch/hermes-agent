import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// The gateway (`ht serve`) the dev UI talks to. Override with HT_GATEWAY_URL.
const GATEWAY = process.env.HT_GATEWAY_URL ?? "http://127.0.0.1:9119";

/**
 * When the gateway enforces auth it injects a one-shot session token into the
 * `index.html` it serves. The Vite dev server serves its own HTML, so scrape
 * the running gateway's token and re-inject it for dev. No-op in prod builds
 * (the Python server injects the token into the built index.html it serves).
 */
function htDevToken(): Plugin {
  const TOKEN_RE =
    /window\.__(?:HT|HERMES)_SESSION_TOKEN__\s*=\s*"([^"]+)"/;
  return {
    name: "ht:dev-session-token",
    apply: "serve",
    async transformIndexHtml(html) {
      try {
        const res = await fetch(GATEWAY, { headers: { accept: "text/html" } });
        const token = (await res.text()).match(TOKEN_RE)?.[1];
        if (token) {
          return html.replace(
            "</head>",
            `<script>window.__HT_SESSION_TOKEN__=${JSON.stringify(token)}</script></head>`,
          );
        }
      } catch {
        // Gateway not running in dev — the app falls back to no-auth loopback.
      }
      return html;
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), htDevToken()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
      "@hermes/shared": path.resolve(__dirname, "../shared/src/index.ts"),
    },
  },
  // Built SPA is served by the Python gateway; keep asset paths relative so it
  // works under a reverse-proxy subpath too.
  base: "./",
  build: { outDir: "dist" },
  server: {
    proxy: {
      "/api": { target: GATEWAY, changeOrigin: true, ws: true },
    },
  },
});
