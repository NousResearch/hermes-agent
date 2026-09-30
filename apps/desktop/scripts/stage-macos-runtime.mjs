// Build-time only, macOS. Supply a standalone Python distribution and a clean,
// tested dependency environment; never point this at a user's Hermes home.
// Output layout (build/runtime) is what electron/bundled-runtime.ts expects:
//   agent/ (tracked source)  python/ (bin/python3 + site-packages)  node/ (optional)  manifest.json
import fs from 'node:fs'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const repo = path.resolve(desktop, '../..')
const options = Object.fromEntries(
  process.argv.slice(2).map(arg => {
    const i = arg.indexOf('=')
    return [arg.slice(0, i), arg.slice(i + 1)]
  })
)
const pythonRoot = options['--python-root']
const sitePackages = options['--site-packages']
const nodeRoot = options['--node-root']
if (process.platform !== 'darwin' || !pythonRoot || !sitePackages) {
  throw new Error(
    'Run on macOS: node scripts/stage-macos-runtime.mjs --python-root=<standalone Python> --site-packages=<clean venv/lib/python3.X/site-packages> [--node-root=<Node distribution>] [--output=<dir>]'
  )
}
if (!fs.existsSync(path.join(pythonRoot, 'bin', 'python3')) || !fs.existsSync(path.join(sitePackages, 'fastapi'))) {
  throw new Error('Expected a standalone Python distribution and tested backend dependencies')
}
const destination = path.resolve(options['--output'] || path.join(desktop, 'build', 'runtime'))
const staging = `${destination}-stage-${Date.now()}`
const agent = path.join(staging, 'agent')
const python = path.join(staging, 'python')
fs.mkdirSync(agent, { recursive: true })
const runtimeDirs = new Set([
  'agent',
  'hermes_cli',
  'tools',
  'gateway',
  'tui_gateway',
  'cron',
  'plugins',
  'skills',
  'optional-skills',
  'acp_adapter',
  'assets',
  'locales',
  'providers',
  'optional-mcps',
  'plugin-catalog',
  'scripts',
  'native'
])
const tracked = execFileSync('git', ['ls-files', '-z'], { cwd: repo, encoding: 'utf8' }).split('\0').filter(Boolean)
for (const name of tracked) {
  const parts = name.split('/')
  const include =
    runtimeDirs.has(parts[0]) ||
    (parts.length === 1 && /\.(py|toml|yaml|json|md|txt)$/.test(name)) ||
    name === 'LICENSE'
  if (!include || parts.some(part => part === '.env' || part === '__pycache__' || part === 'node_modules')) {
    continue
  }
  const target = path.join(agent, name)
  fs.mkdirSync(path.dirname(target), { recursive: true })
  fs.copyFileSync(path.join(repo, name), target)
}
const filter = name => !['__pycache__', '.env'].includes(path.basename(name)) && !name.endsWith('.pyc')
// dereference would turn python3 -> python3.11 into two copies; relative links stay valid inside the tree.
fs.cpSync(pythonRoot, python, { recursive: true, verbatimSymlinks: true, filter })
const libDir = path.join(python, 'lib')
const pyLib = fs.readdirSync(libDir).find(name => /^python3\.\d+$/.test(name))
if (!pyLib) {
  throw new Error('Python distribution has no lib/python3.X directory')
}
const packages = path.join(libDir, pyLib, 'site-packages')
fs.cpSync(sitePackages, packages, { recursive: true, dereference: true, filter })
// Editable installs point into the build machine. Source is supplied through
// PYTHONPATH at runtime; preserve distribution metadata and third-party licenses.
for (const name of fs.readdirSync(packages)) {
  if (name.startsWith('__editable__')) {
    fs.rmSync(path.join(packages, name), { recursive: true, force: true })
  }
  if (name.endsWith('.dist-info')) {
    fs.rmSync(path.join(packages, name, 'direct_url.json'), { force: true })
  }
}
// The PEP 668 marker belongs to a system Python; this interpreter is sealed.
fs.rmSync(path.join(libDir, pyLib, 'EXTERNALLY-MANAGED'), { force: true })

let nodeVersion = null
if (nodeRoot) {
  const node = path.join(staging, 'node')
  fs.mkdirSync(path.join(node, 'bin'), { recursive: true })
  fs.copyFileSync(path.join(nodeRoot, 'bin', 'node'), path.join(node, 'bin', 'node'))
  fs.chmodSync(path.join(node, 'bin', 'node'), 0o755)
  fs.copyFileSync(path.join(nodeRoot, 'LICENSE'), path.join(node, 'LICENSE'))
  fs.cpSync(path.join(nodeRoot, 'lib', 'node_modules', 'npm'), path.join(node, 'lib', 'node_modules', 'npm'), {
    recursive: true,
    dereference: true
  })
  for (const tool of ['npm', 'npx']) {
    fs.symlinkSync(`../lib/node_modules/npm/bin/${tool}-cli.js`, path.join(node, 'bin', tool))
  }
  nodeVersion = execFileSync(path.join(node, 'bin', 'node'), ['--version'], { encoding: 'utf8', timeout: 30000 }).trim()
}

const executable = path.join(python, 'bin', 'python3')
const env = {
  ...process.env,
  PYTHONPATH: agent,
  PYTHONHOME: '',
  VIRTUAL_ENV: '',
  PYTHONNOUSERSITE: '1',
  PYTHONDONTWRITEBYTECODE: '1'
}
// Same reasoning as the Windows stage: the sealed runtime cannot lazy-install
// edge-tts, so read-aloud would fail on first use. Keep the pin in lockstep with
// pyproject.toml's `edge-tts` extra and tools/lazy_deps.py (`tts.edge`).
const EDGE_TTS_VERSION = '7.2.7'
if (!fs.existsSync(path.join(packages, 'edge_tts'))) {
  execFileSync(
    executable,
    ['-m', 'pip', 'install', '--no-input', '--break-system-packages', `edge-tts==${EDGE_TTS_VERSION}`],
    { env, cwd: agent, encoding: 'utf8', timeout: 300000 }
  )
}
if (!fs.existsSync(path.join(packages, 'edge_tts'))) {
  throw new Error('edge-tts is missing from the staged runtime after install; bundled read-aloud would fail')
}
const inventory = JSON.parse(
  execFileSync(
    executable,
    [
      '-c',
      'import json,platform,importlib.metadata as m; import hermes_cli.main,run_agent,model_tools,fastapi,uvicorn,openai,psutil,ptyprocess,edge_tts; print(json.dumps({"python":platform.python_version(),"machine":platform.machine(),"packages":sorted([{ "name":d.metadata["Name"],"version":d.version} for d in m.distributions()],key=lambda d:d["name"].lower())}))'
    ],
    { env, cwd: agent, encoding: 'utf8', timeout: 120000 }
  )
)
const expectedMachine = process.arch === 'arm64' ? 'arm64' : 'x86_64'
if (inventory.machine !== expectedMachine) {
  throw new Error(`Python distribution is ${inventory.machine} but the build targets ${expectedMachine}`)
}
const manifest = {
  schemaVersion: 1,
  platform: 'darwin',
  arch: process.arch,
  commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).trim(),
  ...inventory,
  node: nodeVersion
}
fs.writeFileSync(path.join(staging, 'manifest.json'), JSON.stringify(manifest, null, 2))
// Keep the previous build until the newly staged runtime has passed validation.
if (fs.existsSync(destination)) {
  fs.renameSync(destination, `${destination}-previous-${Date.now()}`)
}
fs.renameSync(staging, destination)
console.log(`Staged ${inventory.packages.length} packages with Python ${inventory.python}: ${destination}`)
