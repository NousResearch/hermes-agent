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

test('every direct E2E spec config/env writer pairing carries its exact mock URL', () => {
  let pairs = 0

  for (const file of fs.readdirSync(import.meta.dirname).filter(name => name.endsWith('.spec.ts'))) {
    const source = fs.readFileSync(path.join(import.meta.dirname, file), 'utf8')
    const ast = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true)
    const calls: ts.CallExpression[] = []

    const visit = (node: ts.Node) => {
      if (
        ts.isCallExpression(node) &&
        ['writeMockProviderConfig', 'writeEnvFile'].includes(node.expression.getText(ast))
      ) {
        calls.push(node)
      }

      ts.forEachChild(node, visit)
    }

    visit(ast)

    for (const env of calls.filter(node => node.expression.getText(ast) === 'writeEnvFile')) {
      const home = env.arguments[0].getText(ast)

      const config = calls
        .filter(
          node =>
            node.expression.getText(ast) === 'writeMockProviderConfig' &&
            node.pos < env.pos &&
            node.arguments[0].getText(ast) === home
        )
        .at(-1)

      assert.ok(config, `${file}: env writer has a preceding same-home config owner`)
      assert.equal(
        env.arguments[2]?.getText(ast),
        config.arguments[1].getText(ast),
        `${file}: env uses the config's exact mock URL`
      )
      pairs += 1
    }
  }

  assert.ok(pairs > 0, 'direct spec pairings are exercised')
})

test('mock-server dev launch writes the credential required by its config before launch', () => {
  const source = fs.readFileSync(new URL('../../../tests-js/scripts/mock-server.ts', import.meta.url), 'utf8')

  const check = (text: string) => {
    const ast = ts.createSourceFile('mock-server.ts', text, ts.ScriptTarget.Latest, true)
    const launch = ast.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === 'runDevLaunch')
    assert.ok(launch, 'actual dev launch function exists')
    const calls: ts.CallExpression[] = []

    const visit = (node: ts.Node) => {
      if (
        ts.isCallExpression(node) &&
        ['writeMockProviderConfig', 'writeEnvFile'].includes(node.expression.getText(ast))
      ) {
        calls.push(node)
      }

      ts.forEachChild(node, visit)
    }

    visit(launch)
    const config = calls.find(node => node.expression.getText(ast) === 'writeMockProviderConfig')
    const env = calls.find(node => node.expression.getText(ast) === 'writeEnvFile')
    assert.ok(config && env)
    assert.equal(env.arguments[0].getText(ast), config.arguments[0].getText(ast))
    assert.equal(env.arguments[2]?.getText(ast), config.arguments[1].getText(ast))
    assert.ok(config.pos < env.pos)
  }

  check(source)

  // A source-only negative oracle proves the old call is rejected without
  // mutating source or launching the development app.
  const legacy = source.replace(
    "writeEnvFile(sandbox.hermesHome, 'e2e-mock-key', mock.url)",
    'writeEnvFile(sandbox.hermesHome)'
  )

  assert.notEqual(legacy, source)
  assert.throws(() => check(legacy))
})
