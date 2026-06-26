'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')

const { findUnexpectedPackagedMainRequires } = require('./packaged-main-validation.cjs')

test('packaged main validator allows bundled externals and builtins', () => {
  const source = `
    const fs = require('node:fs')
    const path = require('path')
    const electron = require('electron')
    const nodePty = require('node-pty')
  `

  assert.deepEqual(findUnexpectedPackagedMainRequires(source), [])
})

test('packaged main validator rejects relative helper requires', () => {
  const source = `
    const helper = require('./backend-probes.cjs')
    const parent = require("../other.cjs")
  `

  assert.deepEqual(findUnexpectedPackagedMainRequires(source), ['../other.cjs', './backend-probes.cjs'])
})

test('packaged main validator rejects unbundled bare package requires', () => {
  const source = `
    const simpleGit = require('simple-git')
    const nested = require("@scope/pkg")
  `

  assert.deepEqual(findUnexpectedPackagedMainRequires(source), ['@scope/pkg', 'simple-git'])
})
