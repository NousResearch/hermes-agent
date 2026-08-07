import fs from 'node:fs'
import path from 'node:path'

function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`
}

function isWithin(child, parent) {
  const relative = path.relative(parent, child)
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))
}

function buildDesktopUpdateArgs(
  branchArgs: string[] = [],
  options: { stageRoot?: string } = {},
) {
  if (options.stageRoot) {
    return ['desktop', '--build-only', '--output-dir', options.stageRoot]
  }
  return ['update', '--yes', '--skip-desktop-build', ...branchArgs]
}

function regexEscape(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function resolveStagedMacApp(stageRoot, exists = fs.existsSync) {
  if (!stageRoot || !path.isAbsolute(stageRoot)) {
    return null
  }
  return [
    path.join(stageRoot, 'mac-arm64', 'Hermes.app'),
    path.join(stageRoot, 'mac', 'Hermes.app')
  ].find(candidate => exists(candidate)) || null
}

function validateMacUpdatePaths({ stageRoot, candidateApp, targetApp }) {
  if (![stageRoot, candidateApp, targetApp].every(value => value && path.isAbsolute(value))) {
    throw new Error('macOS update paths must be absolute')
  }
  const stage = fs.realpathSync(stageRoot)
  const candidate = fs.realpathSync(candidateApp)
  const target = fs.realpathSync(targetApp)
  const next = `${target}.hermes-update-new`
  const previous = `${target}.hermes-update-old`

  if (!isWithin(candidate, stage)) {
    throw new Error('candidate app must be inside staging root')
  }
  if (candidate === target) {
    throw new Error('candidate and target app must be distinct')
  }
  if (isWithin(target, stage) || isWithin(next, stage) || isWithin(previous, stage)) {
    throw new Error('staging root must not contain target or rollback paths')
  }
  return { stageRoot: stage, candidateApp: candidate, targetApp: target, nextApp: next, previousApp: previous }
}

function buildMacPublishScript({
  pid,
  stageRoot,
  candidateApp,
  targetApp,
  waitIterations = 240,
  waitSeconds = 0.5,
  tools = {},
}: {
  pid: number
  stageRoot: string
  candidateApp: string
  targetApp: string
  waitIterations?: number
  waitSeconds?: number
  tools?: {
    copy?: string
    open?: string
    xattr?: string
    validate?: string | null
    startupProbe?: string | null
  }
}) {
  const safe = validateMacUpdatePaths({ stageRoot, candidateApp, targetApp })
  const copy = tools.copy || '/usr/bin/ditto'
  const open = tools.open || '/usr/bin/open'
  const xattr = tools.xattr || '/usr/bin/xattr'
  const validate = tools.validate || null
  const startupProbe = tools.startupProbe || null
  const validateBody = validate
    ? `${shellQuote(validate)} "$1"`
    : '[ -x "$1/Contents/MacOS/Hermes" ]'
  const targetReaderPattern = regexEscape(`${safe.targetApp}${path.sep}`)
  const targetExecutable = path.join(safe.targetApp, 'Contents', 'MacOS', 'Hermes')
  const targetExecutablePattern = regexEscape(targetExecutable)
  const startupProbeBody = startupProbe
    ? `${shellQuote(startupProbe)} "$1"`
    : `/usr/bin/pgrep -f ${shellQuote(targetExecutablePattern)} >/dev/null 2>&1`

  return `#!/bin/bash
set -u
APP_PID=${Number(pid)}
STAGE=${shellQuote(safe.stageRoot)}
SRC=${shellQuote(safe.candidateApp)}
DST=${shellQuote(safe.targetApp)}
NEW=${shellQuote(safe.nextApp)}
OLD=${shellQuote(safe.previousApp)}
EXE=${shellQuote(targetExecutable)}
validate_app() {
  ${validateBody}
}
bundle_readers_alive() {
  if [ "$APP_PID" -gt 0 ] && kill -0 "$APP_PID" 2>/dev/null; then
    return 0
  fi
  /usr/bin/pgrep -f ${shellQuote(targetReaderPattern)} >/dev/null 2>&1
}
startup_ok() {
  ${startupProbeBody}
}
rollback() {
  /bin/rm -rf -- "$DST" 2>/dev/null || true
  if [ "$HAD_TARGET" -eq 1 ] && [ -e "$OLD" ]; then
    /bin/mv -- "$OLD" "$DST" 2>/dev/null || true
  fi
  /bin/rm -rf -- "$NEW" 2>/dev/null || true
}
direct_launch() {
  # LaunchServices 'open' can return success without a durable process.
  # Fall back to the packaged executable under nohup when probes stay dark.
  if [ -x "$EXE" ]; then
    /usr/bin/nohup "$EXE" >/dev/null 2>&1 &
    return 0
  fi
  return 1
}
for _ in $(seq 1 ${Math.max(1, Number(waitIterations) || 1)}); do
  bundle_readers_alive || break
  sleep ${Math.max(0, Number(waitSeconds) || 0)}
done
# Fail closed: no write under the live target or its siblings until the old
# Electron process and every helper reading inside its bundle have exited.
if bundle_readers_alive; then
  exit 20
fi
if [ "$SRC" = "$DST" ] || ! validate_app "$SRC"; then
  exit 21
fi
/bin/rm -rf -- "$NEW" "$OLD" 2>/dev/null || exit 22
if ! ${shellQuote(copy)} "$SRC" "$NEW"; then
  /bin/rm -rf -- "$NEW" 2>/dev/null || true
  exit 23
fi
if ! validate_app "$NEW"; then
  /bin/rm -rf -- "$NEW" 2>/dev/null || true
  exit 24
fi
HAD_TARGET=0
if [ -e "$DST" ]; then
  HAD_TARGET=1
  if ! /bin/mv -- "$DST" "$OLD"; then
    /bin/rm -rf -- "$NEW" 2>/dev/null || true
    exit 25
  fi
fi
if ! /bin/mv -- "$NEW" "$DST"; then
  rollback
  exit 26
fi
if ! validate_app "$DST"; then
  rollback
  exit 27
fi
${shellQuote(xattr)} -dr com.apple.quarantine "$DST" 2>/dev/null || true
if ! ${shellQuote(open)} "$DST"; then
  if ! direct_launch; then
    rollback
    ${shellQuote(open)} "$DST" >/dev/null 2>&1 || true
    exit 28
  fi
fi
# Prefer LaunchServices first; if the process never appears, launch EXE directly.
STABLE=0
DIRECT_TRIED=0
for i in $(seq 1 60); do
  if startup_ok "$DST"; then
    STABLE=$((STABLE + 1))
    if [ "$STABLE" -ge 6 ]; then
      break
    fi
  else
    STABLE=0
    if [ "$DIRECT_TRIED" -eq 0 ] && [ "$i" -ge 6 ]; then
      DIRECT_TRIED=1
      direct_launch || true
    fi
  fi
  sleep 0.5
done
if [ "$STABLE" -lt 6 ]; then
  rollback
  ${shellQuote(open)} "$DST" >/dev/null 2>&1 || true
  exit 29
fi
/bin/rm -rf -- "$OLD" 2>/dev/null || true
# Path validation above proves STAGE cannot contain DST/NEW/OLD, so this exact
# transaction cleanup cannot remove the live or rollback bundle.
/bin/rm -rf -- "$STAGE" 2>/dev/null || true
`
}

export {
  buildDesktopUpdateArgs,
  buildMacPublishScript,
  resolveStagedMacApp,
  shellQuote,
  validateMacUpdatePaths
}
