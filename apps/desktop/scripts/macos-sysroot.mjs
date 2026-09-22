// Build-time only: resolves the macOS SDK the native helpers compile against.
import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'

// `xcrun --sdk macosx` resolves to the highest-versioned SDK installed, not the
// one the active toolchain ships with. A host carrying an SDK newer than its
// Command Line Tools then links against .tbd stubs whose architectures the
// linker cannot parse (#113708). The MacOSX.sdk symlink under the active
// developer dir names the paired SDK, so target that instead.
export function macosSysroot(env = process.env) {
  if (env.SDKROOT) return env.SDKROOT
  const developerDir = execFileSync('xcode-select', ['-p'], { encoding: 'utf8' }).trim()
  return [
    resolve(developerDir, 'SDKs/MacOSX.sdk'),
    resolve(developerDir, 'Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk'),
  ].find(existsSync) ?? null
}

// clang needs the SDK named before the linker runs; when no paired SDK exists
// fall back to xcrun's own default, which at least keeps driver and linker in
// agreement. `--sdk macosx` must precede the tool name or xcrun hands it to clang.
export function xcrunClangArgv(sysroot) {
  return sysroot ? ['clang', '-isysroot', sysroot] : ['--sdk', 'macosx', 'clang']
}
