// Build-time only. Supply a clean, tested Python distribution and dependency
// environment; never point this at a user's Hermes home or copy profile data.
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
const gitRoot = options['--git-root']
const nodeRoot = options['--node-root']
if (process.platform !== 'win32' || !pythonRoot || !sitePackages || !gitRoot || !nodeRoot) {
  throw new Error(
    'Run on Windows: node scripts/stage-windows-runtime.mjs --python-root=<standalone Python> --site-packages=<clean venv/Lib/site-packages> --git-root=<Git distribution> --node-root=<Node distribution>'
  )
}
if (!fs.existsSync(path.join(pythonRoot, 'python.exe')) || !fs.existsSync(path.join(sitePackages, 'fastapi'))) {
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
fs.cpSync(pythonRoot, python, { recursive: true, dereference: true, filter })
const packages = path.join(python, 'Lib', 'site-packages')
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
// Git Bash is needed by Windows terminal tools. Copy distribution files only;
// replace host-specific system configuration and never copy user configuration.
fs.cpSync(gitRoot, path.join(staging, 'git'), {
  recursive: true,
  dereference: true,
  filter: name => filter(name) && !/^unins/i.test(path.basename(name)) && path.basename(name) !== 'gitconfig'
})
fs.mkdirSync(path.join(staging, 'git', 'etc'), { recursive: true })
fs.writeFileSync(
  path.join(staging, 'git', 'etc', 'gitconfig'),
  '[core]\nlongpaths = true\n[http]\nsslBackend = schannel\n'
)
const node = path.join(staging, 'node')
fs.mkdirSync(node, { recursive: true })
for (const name of ['node.exe', 'npm', 'npm.cmd', 'npx', 'npx.cmd', 'LICENSE']) {
  const source = path.join(nodeRoot, name)
  if (fs.existsSync(source)) fs.copyFileSync(source, path.join(node, name))
}
const nodeLicense = options['--node-license'] || path.join(nodeRoot, 'LICENSE')
fs.copyFileSync(nodeLicense, path.join(node, 'LICENSE'))
fs.cpSync(path.join(nodeRoot, 'node_modules', 'npm'), path.join(node, 'node_modules', 'npm'), {
  recursive: true,
  dereference: true
})
for (const binary of [path.join(staging, 'git', 'bin', 'bash.exe'), path.join(node, 'node.exe')]) {
  execFileSync(binary, ['--version'], { encoding: 'utf8', windowsHide: true, timeout: 30000 })
}
const executable = path.join(python, 'python.exe')
const env = {
  ...process.env,
  PYTHONPATH: agent,
  PYTHONHOME: '',
  VIRTUAL_ENV: '',
  PYTHONNOUSERSITE: '1',
  PYTHONDONTWRITEBYTECODE: '1'
}
// Read-aloud ships enabled with Edge as the default TTS provider, but edge-tts
// is an opt-in extra reached at runtime through `tools/lazy_deps.py` (`tts.edge`).
// The bundled runtime is sealed — offline and without a writable install path —
// so that lazy install can never run inside it, and the app fails with
// "No TTS provider available" on the first read-aloud. Bake it in here, exactly
// as the nix `full` image (extraDependencyGroups) and the Docker image (`--extra`)
// already do. The pin MUST stay in lockstep with pyproject.toml's `edge-tts`
// extra and lazy_deps.py's `tts.edge`; bump all three together.
const EDGE_TTS_VERSION = '7.2.7'
if (!fs.existsSync(path.join(packages, 'edge_tts'))) {
  execFileSync(
    executable,
    [
      '-m',
      'pip',
      'install',
      '--no-input',
      '--no-warn-script-location',
      // The standalone distribution ships the PEP 668 EXTERNALLY-MANAGED
      // marker; this interpreter is the sealed runtime, not a system Python,
      // so the override is the intended install path.
      '--break-system-packages',
      `edge-tts==${EDGE_TTS_VERSION}`
    ],
    { env, cwd: agent, encoding: 'utf8', windowsHide: true, timeout: 300000 }
  )
}
if (!fs.existsSync(path.join(packages, 'edge_tts'))) {
  throw new Error(
    'edge-tts is missing from the staged runtime after install; bundled read-aloud would fail'
  )
}
const inventory = JSON.parse(
  execFileSync(
    executable,
    [
      '-c',
      'import json,platform,importlib.metadata as m; import hermes_cli.main,run_agent,model_tools,fastapi,uvicorn,openai,psutil,edge_tts; print(json.dumps({"python":platform.python_version(),"machine":platform.machine(),"packages":sorted([{ "name":d.metadata["Name"],"version":d.version} for d in m.distributions()],key=lambda d:d["name"].lower())}))'
    ],
    { env, cwd: agent, encoding: 'utf8', windowsHide: true, timeout: 120000 }
  )
)
if (
  (process.arch === 'x64' && inventory.machine.toLowerCase() !== 'amd64') ||
  (process.arch === 'arm64' && inventory.machine.toLowerCase() !== 'arm64')
) {
  throw new Error('Python distribution architecture does not match the build host')
}
const manifest = {
  schemaVersion: 1,
  platform: 'win32',
  arch: process.arch,
  commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).trim(),
  ...inventory,
  node: execFileSync(path.join(node, 'node.exe'), ['--version'], { encoding: 'utf8', windowsHide: true }).trim(),
  git: execFileSync(path.join(staging, 'git', 'cmd', 'git.exe'), ['--version'], {
    encoding: 'utf8',
    windowsHide: true
  }).trim()
}
fs.writeFileSync(path.join(staging, 'manifest.json'), JSON.stringify(manifest, null, 2))
// Preserve the previous build until the newly staged runtime has passed validation.
if (fs.existsSync(destination)) {
  fs.renameSync(destination, `${destination}-previous-${Date.now()}`)
}
fs.renameSync(staging, destination)
console.log(`Staged ${inventory.packages.length} packages with Python ${inventory.python}: ${destination}`)
