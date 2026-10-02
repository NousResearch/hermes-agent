/** Exercise the actual fixture AST with inert launch/server boundaries. */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import yaml from 'js-yaml'
import ts from 'typescript'
import { test } from 'vitest'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'

import { buildAppEnvFromParent } from './fixtures-env'

function fixtureFunctions() {
  const source = fs.readFileSync(new URL('./fixtures.ts', import.meta.url), 'utf8')
  const ast = ts.createSourceFile('fixtures.ts', source, ts.ScriptTarget.Latest, true)
  const names = ['createSandbox', 'buildAppEnv', 'setupMockBackend', 'setupPackagedApp', 'setupDeadBackend']
  const nodes = ast.statements.filter(node => ts.isFunctionDeclaration(node) && names.includes(node.name?.text ?? ''))
  assert.equal(nodes.length, names.length)
  const launched: Record<string, string>[] = []
  let appClosed = 0
  let mockClosed = 0
  const page = {}

  const app = {
    close: async () => {
      appClosed += 1
    },
    firstWindow: async () => page
  }

  const deps = {
    fs,
    os,
    path,
    process: { env: {} },
    REPO_ROOT: path.resolve(import.meta.dirname, '../../..'),
    PACKAGED_BINARY_PATH: 'inert-test-binary',
    packagedBinaryExists: () => true,
    writeEnvFile,
    writeMockProviderConfig,
    buildAppEnvFromParent,
    startMockServer: async () => ({
      url: 'http://127.0.0.1:34567',
      close: async () => {
        mockClosed += 1
      }
    }),
    launchDesktop: async (env: Record<string, string>) => {
      launched.push(env)

      return { app, page }
    },
    _electron: {
      launch: async ({ env }: { env: Record<string, string> }) => {
        launched.push(env)

        return app
      }
    },
    installErrorBannerGuard: () => {}
  }

  const body = nodes.map(node => node.getText(ast).replace(/^export\s+/, '')).join('\n')

  const javascript = ts.transpileModule(body, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None }
  }).outputText

  const functions = new Function(
    ...Object.keys(deps),
    `${javascript}\nreturn {setupMockBackend,setupPackagedApp,setupDeadBackend};`
  )(...Object.values(deps))

  return { functions, launched, counts: () => ({ appClosed, mockClosed }) }
}

for (const name of ['setupMockBackend', 'setupPackagedApp', 'setupDeadBackend']) {
  test(`${name} config-required mock credential exists in its isolated home and cleanup owns all fixture resources`, async () => {
    const f = fixtureFunctions()
    const fixture = await f.functions[name]()
    const root = path.resolve(fixture.sandbox.root)
    assert.ok(root.startsWith(path.resolve(os.tmpdir()) + path.sep), 'cleanup stays inside the fresh test temp root')

    try {
      const config = yaml.load(fs.readFileSync(path.join(fixture.sandbox.hermesHome, 'config.yaml'), 'utf8')) as any
      assert.equal(config.model.provider, 'custom')
      const required = config.custom_providers.find((entry: any) => entry.name === 'Mock').key_env
      assert.equal(required, 'OPENAI_API_KEY')

      const keys = new Set(
        fs
          .readFileSync(path.join(fixture.sandbox.hermesHome, '.env'), 'utf8')
          .split('\n')
          .map(line => line.split('=')[0])
      )

      assert.ok(keys.has(required), 'configured provider credential is written locally without ambient secrets')
      assert.equal(f.launched.length, 1)
      assert.equal(f.launched[0].HERMES_HOME, fixture.sandbox.hermesHome)
      assert.equal(f.launched[0].OPENAI_API_KEY, undefined, 'no ambient credential reaches the launch environment')
    } finally {
      await fixture.cleanup()
      assert.equal(fs.existsSync(root), false)
      assert.deepEqual(f.counts(), { appClosed: 1, mockClosed: name === 'setupDeadBackend' ? 0 : 1 })
    }
  })
}
