import { readFileSync } from 'node:fs'

import assert from 'node:assert/strict'
import ts from 'typescript'
import { test } from 'vitest'

import { resolveProfileBackendRoute } from './connection-config'

const mainSource = readFileSync(new URL('./main.ts', import.meta.url), 'utf8')
const sourceFile = ts.createSourceFile('main.ts', mainSource, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)

function functionDeclaration(name: string): ts.FunctionDeclaration {
  const declaration = sourceFile.statements.find(
    (statement): statement is ts.FunctionDeclaration =>
      ts.isFunctionDeclaration(statement) && statement.name?.text === name
  )

  assert.ok(declaration, `main.ts must declare ${name}`)
  return declaration
}

function property(object: ts.ObjectLiteralExpression, name: string): ts.ObjectLiteralElementLike | undefined {
  return object.properties.find(
    item => ts.isPropertyAssignment(item) && ts.isIdentifier(item.name) && item.name.text === name
  )
}

function propertyValue(object: ts.ObjectLiteralExpression, name: string): ts.Expression | undefined {
  const found = property(object, name)
  return found && ts.isPropertyAssignment(found) ? found.initializer : undefined
}

function callsNamed(root: ts.Node, name: string): ts.CallExpression[] {
  const found: ts.CallExpression[] = []

  function visit(node: ts.Node) {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === name) {
      found.push(node)
    }
    ts.forEachChild(node, visit)
  }

  visit(root)
  return found
}

function requestOption(expression: ts.Expression | undefined): boolean {
  if (!expression || !ts.isObjectLiteralExpression(expression)) return false
  const request = propertyValue(expression, 'request')
  if (!request || !ts.isObjectLiteralExpression(request)) return false
  const method = propertyValue(request, 'method')
  const path = propertyValue(request, 'path')
  return method?.getText(sourceFile) === 'request?.method' && path?.getText(sourceFile) === 'request?.path'
}

test('registry-pinned plugin API request metadata reaches local profile backend routing', () => {
  const dispatch = functionDeclaration('dispatchRegistryApiRequest')
  const registryCalls = callsNamed(dispatch, 'ensureRegistryBackend')
  assert.equal(registryCalls.length, 2, 'active and passive registry dispatch paths must both be covered')

  for (const call of registryCalls) {
    assert.ok(requestOption(call.arguments[3]), 'registry dispatch must forward HTTP method and path')
  }

  const ensureRegistry = functionDeclaration('ensureRegistryBackend')
  const delegatedCalls = callsNamed(ensureRegistry, 'ensureBackend').filter(call => {
    const options = call.arguments[1]
    const forwardedRequest = options && ts.isObjectLiteralExpression(options) ? propertyValue(options, 'request') : undefined
    return Boolean(
      forwardedRequest &&
        ts.isPropertyAccessExpression(forwardedRequest) &&
        ts.isIdentifier(forwardedRequest.expression) &&
        forwardedRequest.expression.text === 'opts' &&
        forwardedRequest.name.text === 'request'
    )
  })
  assert.ok(delegatedCalls.length >= 2, 'local delegate and primary reuse must retain registry request metadata')

  const forcedLocalSpawn = callsNamed(ensureRegistry, 'spawnPoolBackend').find(call => {
    const options = call.arguments[2]
    return options && ts.isObjectLiteralExpression(options) && property(options, 'forceLocal')
  })
  assert.ok(forcedLocalSpawn, 'explicit local routing must keep its forced-local pool')
  const spawnOptions = forcedLocalSpawn.arguments[2]
  assert.ok(spawnOptions && ts.isObjectLiteralExpression(spawnOptions))
  const processScopeGuard = propertyValue(spawnOptions, 'unscopableRequest')
  assert.ok(processScopeGuard && ts.isCallExpression(processScopeGuard))
  assert.equal(processScopeGuard.expression.getText(sourceFile), 'unscopableMutatingRequest')
  const guardArguments = processScopeGuard.arguments[0]
  assert.ok(guardArguments && ts.isObjectLiteralExpression(guardArguments))
  assert.equal(propertyValue(guardArguments, 'requestPath')?.getText(sourceFile), 'opts.request?.path')
  assert.equal(propertyValue(guardArguments, 'requestMethod')?.getText(sourceFile), 'opts.request?.method')

  // This is the route reached once the bridge carries the request metadata
  // above: registry-pinned local plugin reads must not collapse onto mtplx.
  assert.deepEqual(
    resolveProfileBackendRoute('coder', {
      primaryProfile: 'default',
      globalRemote: false,
      profileRemoteOverride: false,
      requestMethod: 'GET',
      requestPath: '/api/plugins/plur1bus/status'
    }),
    { backend: 'pool', descriptorProfile: null, scopePath: false }
  )
})
