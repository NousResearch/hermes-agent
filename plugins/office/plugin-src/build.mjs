/**
 * esbuild build script for the Office dashboard plugin.
 *
 * Everything (React, react-dom, Three.js, R3F, drei) is bundled inline.
 * The plugin does NOT shim React from the dashboard SDK for the 3D tree —
 * instead, OfficeView (rendered by the dashboard's React) creates a div and
 * uses the bundled react-dom/client's createRoot to render the entire 3D
 * scene (Canvas + all children) into that div with the bundled React. This
 * avoids the React instance mismatch that caused R3F's react-reconciler to
 * render nothing (blank/transparent canvas).
 *
 * The dashboard's React only ever touches the thin container div + the
 * 2D fallback / error boundary; the bundled React owns the entire R3F tree.
 */
import * as esbuild from "esbuild";

await esbuild.build({
  entryPoints: ["src/index.tsx"],
  bundle: true,
  format: "iife",
  outfile: "../dashboard/dist/index.js",
  loader: { ".tsx": "tsx" },
  jsx: "automatic",
  jsxImportSource: "react",
  logLevel: "info",
});
