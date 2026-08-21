import { accessSync, readFileSync } from "fs"
import { createRequire } from "module"
import { resolve, join } from "path"

const app = resolve(import.meta.dirname, "..")
const root = resolve(app, "..", "..")

try {
  accessSync(join(root, "node_modules", "vite", "package.json"))
} catch {
  console.error(`Run from repo root: cd ${root} && npm ci`)
  process.exit(1)
}

// `vite.config.ts` aliases react/react-dom to whatever this workspace resolves,
// and React refuses to run when the two come from different installed copies
// ("Minified React error #527" — it throws before the first paint, so the app
// window stays blank). npm stays silent about the split because the hoisted
// react still satisfies react-dom's caret peer range. Fail the build loudly
// instead of shipping a white screen.
const requireFromApp = createRequire(join(app, "package.json"))
const installedVersion = (pkg) =>
  JSON.parse(readFileSync(requireFromApp.resolve(`${pkg}/package.json`), "utf8")).version

const react = installedVersion("react")
const reactDom = installedVersion("react-dom")

if (react !== reactDom) {
  console.error(
    `react@${react} / react-dom@${reactDom} version mismatch — React would fail ` +
      `with error #527 and render a blank window. Pin both to the same version ` +
      `in ${join(app, "package.json")}, then reinstall: cd ${root} && npm ci`
  )
  process.exit(1)
}
