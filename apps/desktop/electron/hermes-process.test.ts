import { describe, expect, it } from 'vitest'

import { commandLineIsHermesRuntime } from './hermes-process'

// HX003: process identity comes from a whole argv TOKEN, never an argv substring.
describe('commandLineIsHermesRuntime (HX003 token matching)', () => {
  it('matches a real Hermes gateway command line', () => {
    expect(commandLineIsHermesRuntime('hermes\0gateway\0run')).toBe(true)
    expect(commandLineIsHermesRuntime('hermes gateway run')).toBe(true)
    expect(commandLineIsHermesRuntime('hermes --profile work gateway run')).toBe(true)
  })

  it('matches a Hermes launched by an interpreter (basename of a path token)', () => {
    expect(commandLineIsHermesRuntime('/venv/bin/python\0/venv/bin/hermes\0gateway\0run')).toBe(true)
    expect(commandLineIsHermesRuntime('C:\\Hermes\\hermes.exe gateway run')).toBe(true)
  })

  // The bug this replaces: a bare `.includes('hermes')` matched any of these.
  it('does NOT match an unrelated process whose path merely contains "hermes"', () => {
    expect(commandLineIsHermesRuntime('vim /home/user/hermes-notes.txt')).toBe(false)
    expect(commandLineIsHermesRuntime('/opt/hermesbrew/bin/tool --run')).toBe(false)
    expect(commandLineIsHermesRuntime('node /srv/hermes-dashboard/server.js')).toBe(false)
    expect(commandLineIsHermesRuntime('cat ~/.hermes/config.yaml')).toBe(false)
  })

  // G2 (#150283): the real unmanaged-gateway launch is `python -m hermes_cli.main … gateway
  // run`, which the executable-basename-only matcher REJECTED — so findRunningGatewayProfiles
  // missed a live gateway at first launch / profile repair. Mirror the canonical Python matcher
  // (gateway.status.looks_like_gateway_command_line): recognize the module entry point, by argv
  // TOKENS not a substring, carrying the gateway subcommand.
  it('matches a module-launched gateway (python -m hermes_cli.main … gateway run)', () => {
    expect(commandLineIsHermesRuntime('python\0-m\0hermes_cli.main\0-p\0x\0gateway\0run')).toBe(true)
    expect(commandLineIsHermesRuntime('python -m hermes_cli.main -p x gateway run')).toBe(true)
    expect(commandLineIsHermesRuntime('/venv/bin/python /venv/lib/hermes_cli/main.py gateway run')).toBe(true)
    expect(commandLineIsHermesRuntime('python -m hermes_cli.main --profile work gateway run')).toBe(true)
  })

  it('does NOT match a module entry without the gateway subcommand, nor an unrelated hermes-tool path', () => {
    // A hermes_cli.main invocation that is not a gateway (e.g. `chat`) is not a running gateway.
    expect(commandLineIsHermesRuntime('python -m hermes_cli.main chat')).toBe(false)
    // HX003: no substring; an unrelated tool whose path contains "hermes" is never a runtime.
    expect(commandLineIsHermesRuntime('/opt/hermes-tool/foo gateway run')).toBe(false)
    // `-p gateway` is a profile value, not a subcommand.
    expect(commandLineIsHermesRuntime('python -m hermes_cli.main -p gateway chat')).toBe(false)
  })
})
