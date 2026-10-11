import net from 'node:net'
import path from 'node:path'
import fs from 'node:fs'
import os from 'node:os'

import { describe, expect, it, vi } from 'vitest'

import {
  assertPortsAvailable,
  assertRehearsalDirectory,
  configureRehearsalApp,
  parseRehearsalArgs,
  prepareRehearsal,
  rehearsalPortChecks
} from './connector-rehearsal.mjs'

describe('connector rehearsal preparation', () => {
  it('cannot register a consent handler or reuse the installed Windows notification identity', () => {
    const registerProtocol = vi.fn()
    const setAppUserModelId = vi.fn()
    const app = { setAsDefaultProtocolClient: registerProtocol, setAppUserModelId, setAppPath: vi.fn() }
    configureRehearsalApp(app, 'C:\\src\\hermes\\apps\\desktop', 'win32')
    expect(app.setAsDefaultProtocolClient('hermes')).toBe(false)
    expect(registerProtocol).not.toHaveBeenCalled()
    app.setAppUserModelId('com.nousresearch.hermes')
    expect(setAppUserModelId).toHaveBeenCalledWith('com.nousresearch.hermes.connector-rehearsal')
    expect(app.setAppPath).toHaveBeenCalledWith('C:\\src\\hermes\\apps\\desktop')
  })
  it('refuses an arbitrary nonempty directory and accepts a previously marked rehearsal', () => {
    const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'connector-rehearsal-directory-'))
    const config = path.join(sandbox, 'existing-profile')
    const marker = path.join(sandbox, '.connector-rehearsal.json')
    try {
      expect(() => assertRehearsalDirectory(sandbox)).not.toThrow()
      fs.writeFileSync(config, 'fixture profile must remain untouched')
      expect(() => assertRehearsalDirectory(sandbox)).toThrow('Choose an empty directory')
      expect(fs.readFileSync(config, 'utf8')).toBe('fixture profile must remain untouched')
      fs.writeFileSync(marker, JSON.stringify({ schema: 1 }))
      expect(() => assertRehearsalDirectory(sandbox)).not.toThrow()
    } finally {
      fs.unlinkSync(config)
      if (fs.existsSync(marker)) fs.unlinkSync(marker)
      fs.rmdirSync(sandbox)
    }
  })
  it('preserves POSIX launch variables and keeps CDP off unless explicitly enabled', () => {
    const base = {
      sandbox: '/tmp/hermes-connectors-test',
      desktop: '/src/hermes/apps/desktop',
      root: '/src/hermes',
      python: '/runtime/bin/python',
      platform: 'linux',
      sourceEnv: {
        PATH: '/usr/bin',
        LANG: 'en_US.UTF-8',
        DISPLAY: ':0',
        XAUTHORITY: '/tmp/xauthority',
        DBUS_SESSION_BUS_ADDRESS: 'unix:path=/run/user/1000/bus',
        OPENAI_API_KEY: 'must-not-pass'
      }
    }
    const prepared = prepareRehearsal({ ...base, cdpPort: parseRehearsalArgs([]).cdpPort })

    expect(prepared.env.PATH).toBe('/usr/bin')
    expect(prepared.env.LANG).toBe('en_US.UTF-8')
    expect(prepared.env.DISPLAY).toBe(':0')
    expect(prepared.env.XAUTHORITY).toBe('/tmp/xauthority')
    expect(prepared.env.DBUS_SESSION_BUS_ADDRESS).toBe('unix:path=/run/user/1000/bus')
    expect(prepared.env).not.toHaveProperty('OPENAI_API_KEY')
    expect(prepared.env.HERMES_DESKTOP_CDP_PORT).toBe('off')
    expect(rehearsalPortChecks(prepared)).toEqual([{ label: 'Renderer', port: 5194 }])

    const explicitPort = parseRehearsalArgs(['--cdp-port', '9344']).cdpPort
    const debugRehearsal = prepareRehearsal({ ...base, cdpPort: explicitPort })
    expect(debugRehearsal.env.HERMES_DESKTOP_CDP_PORT).toBe('9344')
    expect(rehearsalPortChecks(debugRehearsal)).toEqual([
      { label: 'Renderer', port: 5194 },
      { label: 'CDP', port: 9344 }
    ])
    expect(() => parseRehearsalArgs(['--cdp-port', '65536'])).toThrow('CDP port must be')
  })

  it('isolates Windows paths while retaining required OS variables and dropping auth overrides', () => {
    const sandbox = 'C:\\Users\\TestUser\\AppData\\Local\\Temp\\hermes-connectors-test'
    const prepared = prepareRehearsal({
      sandbox,
      desktop: 'C:\\src\\hermes\\apps\\desktop',
      root: 'C:\\src\\hermes',
      python: 'C:\\hermes-runtime\\python.exe',
      platform: 'win32',
      sourceEnv: {
        SystemRoot: 'C:\\Windows',
        windir: 'C:\\Windows',
        Path: 'C:\\Windows\\System32;C:\\Program Files\\nodejs',
        ComSpec: 'C:\\Windows\\System32\\cmd.exe',
        TEMP: 'C:\\Users\\TestUser\\AppData\\Local\\Temp',
        USERPROFILE: 'C:\\Users\\TestUser',
        APPDATA: 'C:\\Users\\TestUser\\AppData\\Roaming',
        LOCALAPPDATA: 'C:\\Users\\TestUser\\AppData\\Local',
        OPENAI_API_KEY: 'must-not-pass',
        AWS_ACCESS_KEY_ID: 'must-not-pass',
        NODE_OPTIONS: '--require=C:\\credential-hook.js',
        HERMES_HOME: 'C:\\Users\\TestUser\\.hermes',
        HERMES_RUNTIME_DIR: 'C:\\Users\\TestUser\\.hermes\\tools',
        HERMES_SHARED_AUTH_DIR: 'C:\\Users\\TestUser\\.hermes\\shared'
      }
    })
    const winPath = path.win32

    expect(prepared.env.SYSTEMROOT).toBe('C:\\Windows')
    expect(prepared.env.WINDIR).toBe('C:\\Windows')
    expect(prepared.env.PATH).toBe('C:\\Windows\\System32;C:\\Program Files\\nodejs')
    expect(prepared.env.COMSPEC).toBe('C:\\Windows\\System32\\cmd.exe')
    expect(prepared.env.USERPROFILE).toBe(sandbox)
    expect(prepared.env.TEMP).toBe(winPath.join(sandbox, 'tmp'))
    expect(prepared.env.TMP).toBe(winPath.join(sandbox, 'tmp'))
    expect(prepared.env.APPDATA).toBe(winPath.join(sandbox, 'AppData', 'Roaming'))
    expect(prepared.env.LOCALAPPDATA).toBe(winPath.join(sandbox, 'AppData', 'Local'))
    expect(prepared.env.HOME).toBe(sandbox)
    expect(prepared.env.HERMES_HOME).toBe(winPath.join(sandbox, '.hermes'))
    expect(prepared.env.HERMES_SHARED_AUTH_DIR).toBe(winPath.join(sandbox, '.hermes', 'shared'))
    expect(prepared.env.HERMES_RUNTIME_DIR).toBe(winPath.join(sandbox, 'runtime'))
    expect(prepared.env.HERMES_DESKTOP_USER_DATA_DIR).toBe(winPath.join(sandbox, 'electron'))
    expect(prepared.env.HERMES_DESKTOP_CWD).toBe(winPath.join(sandbox, 'work'))
    expect(prepared.env.HERMES_DESKTOP_APP_NAME).toBe('Hermes Connector Rehearsal')
    expect(prepared.env.HERMES_DESKTOP_CDP_PORT).toBe('off')
    expect(prepared.url).toBe('http://127.0.0.1:5194')

    for (const key of ['OPENAI_API_KEY', 'AWS_ACCESS_KEY_ID', 'NODE_OPTIONS']) {
      expect(prepared.env).not.toHaveProperty(key)
    }
    for (const dir of [
      prepared.hermesHome,
      prepared.sharedAuthDir,
      prepared.runtimeDir,
      prepared.userDataDir,
      prepared.workingDirectory
    ]) {
      const relative = winPath.relative(sandbox, dir)
      expect(
        relative && !relative.startsWith('..') && !winPath.isAbsolute(relative),
        `${dir} must stay under the sandbox`
      ).toBe(true)
    }
  })

  it.each(['Renderer', 'CDP'])('refuses an occupied %s port', async label => {
    const occupied = net.createServer()
    const port = await new Promise((resolve, reject) => {
      occupied.once('error', reject)
      occupied.listen(0, '127.0.0.1', () => resolve(occupied.address().port))
    })

    try {
      await expect(assertPortsAvailable([{ label, port }])).rejects.toThrow(`${label} port ${port} is unavailable`)
    } finally {
      await new Promise(resolve => occupied.close(resolve))
    }
  })
})
