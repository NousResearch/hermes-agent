// Git Bash/MSYS rewrite slash-prefixed native flags (`/FI`) into paths.
// Set these two opt-outs on Windows unless already present.
export function applyWindowsMsysBashEnvDefaults(
  env: NodeJS.ProcessEnv,
  isWindows: boolean = process.platform === 'win32'
): NodeJS.ProcessEnv {
  if (!isWindows) {
    return env
  }

  if (!Object.prototype.hasOwnProperty.call(env, 'MSYS_NO_PATHCONV')) {
    env.MSYS_NO_PATHCONV = '1'
  }

  if (!Object.prototype.hasOwnProperty.call(env, 'MSYS2_ARG_CONV_EXCL')) {
    env.MSYS2_ARG_CONV_EXCL = '*'
  }

  return env
}
