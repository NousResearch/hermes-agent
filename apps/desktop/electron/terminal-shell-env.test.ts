import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

vi.mock('electron', () => ({ app: {}, ipcMain: {} }))
vi.mock('node-pty', () => ({ default: {} }))

import { scrubBackendPythonEnv } from './terminal-ipc'

test('scrubBackendPythonEnv drops the backend Python activation env from interactive panes', () => {
  const scrubbed = scrubBackendPythonEnv({
    PYTHONPATH: 'C:\\checkout;C:\\checkout\\venv\\Lib\\site-packages',
    PYTHONHOME: 'C:\\checkout\\venv',
    VIRTUAL_ENV: 'C:\\checkout\\venv',
    CONDA_DEFAULT_ENV: 'base',
    CONDA_SHLVL: '1',
    conda_prompt_modifier: '(base)',
    _CE_M: '',
    _CE_CONDA: '',
    PATH: 'C:\\Windows\\system32',
    TERM: 'xterm-256color',
    HERMES_DESKTOP: '1'
  })

  assert.equal(scrubbed.PYTHONPATH, undefined)
  assert.equal(scrubbed.PYTHONHOME, undefined)
  assert.equal(scrubbed.VIRTUAL_ENV, undefined)
  assert.equal(scrubbed.CONDA_DEFAULT_ENV, undefined)
  assert.equal(scrubbed.CONDA_SHLVL, undefined)
  assert.equal(scrubbed.conda_prompt_modifier, undefined)
  assert.equal(scrubbed._CE_M, undefined)
  assert.equal(scrubbed._CE_CONDA, undefined)
  assert.equal(scrubbed.PATH, 'C:\\Windows\\system32')
  assert.equal(scrubbed.TERM, 'xterm-256color')
  assert.equal(scrubbed.HERMES_DESKTOP, '1')
})

test('scrubBackendPythonEnv does not mutate its input', () => {
  const source = { PYTHONPATH: '/leaked', PATH: '/bin' }

  scrubBackendPythonEnv(source)

  assert.equal(source.PYTHONPATH, '/leaked')
})
