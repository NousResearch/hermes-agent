'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { buildCommand, HERMES_SPEC } = require('../bin/hermes-agent-acp.js');

test('uses uvx when available and forwards args', () => {
  const command = buildCommand(['--version'], (name) => name === 'uvx');

  assert.equal(command.command, 'uvx');
  assert.deepEqual(command.args, ['--from', HERMES_SPEC, 'hermes-acp', '--version']);
});

test('falls back to uv tool run and forwards setup args', () => {
  const command = buildCommand(['--setup'], (name) => name === 'uv');

  assert.equal(command.command, 'uv');
  assert.deepEqual(command.args, ['tool', 'run', '--from', HERMES_SPEC, 'hermes-acp', '--setup']);
});

test('returns null when neither uvx nor uv is available', () => {
  assert.equal(buildCommand([], () => false), null);
});
