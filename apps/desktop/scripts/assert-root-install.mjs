import { accessSync } from "fs"
import { resolve, join } from "path"

const root = resolve(import.meta.dirname, "..", "..", "..")

try {
  accessSync(join(root, "node_modules", "vite", "package.json"))
} catch {
  console.error(`Run from repo root: cd ${root} && npm ci`)
  process.exit(1)
}
