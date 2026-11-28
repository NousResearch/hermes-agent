// Build-time only: resolves the macOS SDK the native helpers compile against.
import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'

// Every rung below is a read: a failing tool or a missing directory falls to
// the next rung instead of failing the build.
const read = (file, args, env) => {
  try {
    return execFileSync(file, args, { encoding: 'utf8', env, stdio: 'pipe' }).trim()
  } catch {
    return ''
  }
}

// Every macOS SDK carries SDKSettings.plist; xcrun accepts any existing
// directory as an SDK path, so existence alone does not prove one.
const isSdk = path => Boolean(path) && existsSync(resolve(path, 'SDKSettings.plist'))

// The installed SDK an explicit SDKROOT names, or null. SDKROOT accepts SDK
// names as well as paths; clang's -isysroot only accepts paths, so both forms
// go through xcrun.
function explicitSysroot(sdkroot, env) {
  const path = read('xcrun', ['--sdk', sdkroot, '--show-sdk-path'], env)
  return isSdk(path) ? path : null
}

// The MacOSX.sdk alias of the active developer directory (Command Line Tools
// layout first, then Xcode's), or null when the toolchain has neither.
function developerSysroot(env) {
  const developerDir = read('xcode-select', ['-p'], env)
  if (!developerDir) return null
  return [
    resolve(developerDir, 'SDKs/MacOSX.sdk'),
    resolve(developerDir, 'Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk'),
  ].find(isSdk) ?? null
}

// `--sdk macosx` can select a newer SDK than the linker understands (#113708).
// Prefer the active toolchain's MacOSX.sdk alias unless the caller pins an SDK.
// A stale SDKROOT never fails the build: it is reported, then ignored, which
// matches what `xcrun --sdk macosx` did with it before this resolver existed.
export function macosSysroot(env = process.env) {
  if (env.SDKROOT) {
    const explicit = explicitSysroot(env.SDKROOT, env)
    if (explicit) return explicit
    console.warn(`[macos-sysroot] SDKROOT=${env.SDKROOT} is not an installed SDK; using the active toolchain's default`)
  }
  return developerSysroot(env)
}

// xcrun looks clang up through an SDK before clang ever sees -isysroot, so an
// inherited stale SDKROOT fails that lookup: always name the SDK, before the
// tool name or xcrun passes `--sdk` to clang. Without a sysroot rung,
// `--sdk macosx` keeps xcrun's previous default selection.
export function xcrunClangArgv(sysroot) {
  return sysroot ? ['--sdk', sysroot, 'clang', '-isysroot', sysroot] : ['--sdk', 'macosx', 'clang']
}
