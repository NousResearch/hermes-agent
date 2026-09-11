import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { provisionCliLinks } from './cli-provision'

function fixture(): { root: string; binDir: string; source: string } {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-cli-links-'))
  const binDir = path.join(root, 'bin')
  const source = path.join(root, 'new', 'agent-payload', 'bin', 'hermes')

  fs.mkdirSync(path.dirname(source), { recursive: true })
  fs.mkdirSync(binDir)
  fs.writeFileSync(source, 'payload command\n')

  return { root, binDir, source }
}

test('repairs owned dangling CLI links without changing foreign or live entries', context => {
  const { root, binDir, source } = fixture()
  const messages: string[] = []
  const target = path.join(binDir, 'hermes')

  try {
    const oldSource = path.join(root, 'old', 'agent-payload', 'bin', 'hermes')

    try {
      fs.symlinkSync(oldSource, target)
    } catch (error) {
      if (process.platform === 'win32' && (error as NodeJS.ErrnoException).code === 'EPERM') {
        context.skip('Windows symlinks require Developer Mode or elevation')
      }

      throw error
    }

    provisionCliLinks({ hermes: source }, binDir, message => messages.push(message))
    assert.equal(fs.readlinkSync(target), source)
    assert.equal(fs.readFileSync(source, 'utf8'), 'payload command\n')
    assert.deepEqual(fs.readdirSync(binDir), ['hermes'])

    const foreign = path.join(root, 'removed-other-tool', 'hermes')
    const otherName = path.join(root, 'old', 'agent-payload', 'bin', 'other')

    for (const destination of [source, foreign, otherName, 'agent-payload/bin/hermes']) {
      fs.unlinkSync(target)
      fs.symlinkSync(destination, target)
      const original = fs.readlinkSync(target)

      provisionCliLinks({ hermes: source }, binDir, message => messages.push(message))
      assert.equal(fs.readlinkSync(target), original)
      assert.deepEqual(fs.readdirSync(binDir), ['hermes'])
    }

    fs.unlinkSync(target)
    fs.writeFileSync(target, 'user command\n')
    provisionCliLinks({ hermes: source }, binDir, message => messages.push(message))
    assert.equal(fs.readFileSync(target, 'utf8'), 'user command\n')
    fs.unlinkSync(target)
    provisionCliLinks({ hermes: source }, binDir, message => messages.push(message))
    assert.equal(fs.readlinkSync(target), source)
    assert.equal(messages.filter(message => message.includes('linked 1')).length, 2)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('qualified CLI paths expose their filenames, not shared canonical command keys', context => {
  const { root, binDir, source } = fixture()

  try {
    const plain = path.join(binDir, 'hermes')
    fs.writeFileSync(plain, 'stable command')

    try {
      fs.symlinkSync(source, path.join(binDir, 'probe'))
      fs.unlinkSync(path.join(binDir, 'probe'))
    } catch (error) {
      if (process.platform === 'win32' && (error as NodeJS.ErrnoException).code === 'EPERM') {
        context.skip('Windows symlinks require Developer Mode or elevation')
      }

      throw error
    }

    for (const name of ['hermes-canary', 'hermes-abcdef1', 'hermes-1234567']) {
      const cli = path.join(path.dirname(source), name)
      const acp = `${cli}-acp`
      fs.writeFileSync(cli, name)
      fs.writeFileSync(acp, `${name}-acp`)
      provisionCliLinks({ hermes: cli, 'hermes-acp': acp }, binDir, () => {})
      assert.equal(fs.readlinkSync(path.join(binDir, name)), cli)
      assert.equal(fs.readlinkSync(path.join(binDir, `${name}-acp`)), acp)
    }

    assert.equal(fs.readFileSync(plain, 'utf8'), 'stable command')
    assert.equal(fs.existsSync(path.join(binDir, 'hermes-acp')), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('a failed link swap preserves its source and target, then provisions later commands', context => {
  const { root, binDir, source } = fixture()
  const target = path.join(binDir, 'hermes')
  const oldSource = path.join(root, 'old', 'agent-payload', 'bin', 'hermes')
  const acpSource = path.join(path.dirname(source), 'hermes-acp')
  const messages: string[] = []
  let swaps = 0

  try {
    try {
      fs.symlinkSync(oldSource, target)
    } catch (error) {
      if (process.platform === 'win32' && (error as NodeJS.ErrnoException).code === 'EPERM') {
        context.skip('Windows symlinks require Developer Mode or elevation')
      }

      throw error
    }

    fs.writeFileSync(acpSource, 'ACP command\n')
    provisionCliLinks({ hermes: source, 'hermes-acp': acpSource }, binDir, message => messages.push(message), {
      ...fs,
      renameSync: (from, to) => {
        swaps += 1
        fs.unlinkSync(from)
        fs.renameSync(from, to)
      }
    })

    assert.equal(swaps, 1)
    assert.equal(fs.readlinkSync(target), oldSource)
    assert.equal(fs.readFileSync(source, 'utf8'), 'payload command\n')
    assert.equal(fs.readlinkSync(path.join(binDir, 'hermes-acp')), acpSource)
    assert.deepEqual(fs.readdirSync(binDir).sort(), ['hermes', 'hermes-acp'])
    assert.ok(messages.some(message => message.includes('hermes') && message.includes('ENOENT')))
    assert.ok(messages.some(message => message.includes('linked 1')))
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
