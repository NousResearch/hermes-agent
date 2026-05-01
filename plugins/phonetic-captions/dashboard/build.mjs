/**
 * Build script for phonetic-captions dashboard plugin.
 *
 * Produces an IIFE bundle at dist/index.js that:
 * - Uses React from the host SDK (window.__HERMES_PLUGIN_SDK__.React)
 * - Bundles lucide-react icons directly
 * - Registers with window.__HERMES_PLUGINS__
 */
import { build } from "esbuild";
import { writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Create a shim for react that reads from the SDK global
const shimDir = join(__dirname, "node_modules", ".shims");
mkdirSync(shimDir, { recursive: true });

writeFileSync(
  join(shimDir, "react.js"),
  `const R = window.__HERMES_PLUGIN_SDK__.React;
module.exports = R;
module.exports.default = R;
module.exports.useState = R.useState;
module.exports.useEffect = R.useEffect;
module.exports.useCallback = R.useCallback;
module.exports.useMemo = R.useMemo;
module.exports.useRef = R.useRef;
module.exports.createElement = R.createElement;
module.exports.Fragment = R.Fragment;
`
);

writeFileSync(
  join(shimDir, "react-jsx-runtime.js"),
  `const R = window.__HERMES_PLUGIN_SDK__.React;
module.exports.jsx = R.createElement;
module.exports.jsxs = R.createElement;
module.exports.Fragment = R.Fragment;
`
);

await build({
  entryPoints: [join(__dirname, "src", "index.tsx")],
  bundle: true,
  format: "iife",
  outfile: join(__dirname, "dist", "index.js"),
  minify: true,
  target: "es2020",
  jsx: "automatic",
  alias: {
    react: join(shimDir, "react.js"),
    "react/jsx-runtime": join(shimDir, "react-jsx-runtime.js"),
    "react-dom": join(shimDir, "react.js"),
  },
  define: {
    "process.env.NODE_ENV": '"production"',
  },
  logLevel: "info",
});

console.log("✓ Built dist/index.js");
