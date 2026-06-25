"use strict"

const fs = require("fs")
const path = require("path")

const root = path.resolve(__dirname, "..", "..", "..")
const desktopDir = path.resolve(__dirname, "..")

// Vite is a devDependency of apps/desktop. With npm workspaces it lives in
// apps/desktop/node_modules (or is hoisted to the repo root only when another
// workspace depends on the same version). Check both locations so the guard
// works for both the standalone and the workspace-hoisted install layouts.
const candidates = [
  path.join(desktopDir, "node_modules", "vite", "package.json"),
  path.join(root, "node_modules", "vite", "package.json"),
]

if (candidates.some((p) => fs.existsSync(p))) {
  process.exit(0)
}

console.error(
  `Desktop workspace dependencies are not installed.\n` +
  `Run from repo root: cd ${root} && npm install\n` +
  `If apps/desktop/node_modules is missing, also run: cd ${desktopDir} && npm install`
)
process.exit(1)
