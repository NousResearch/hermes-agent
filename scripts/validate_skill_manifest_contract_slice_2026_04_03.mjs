import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { validateSkillFile } from '../tools/skill_manifest_contract/validator.mjs'

const execFileAsync = promisify(execFile)

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

async function checkSyntax(filePath) {
  await execFileAsync('node', ['--check', filePath])
}

function issueCodes(result) {
  return new Set(result.contract.issues.map((issue) => issue.code))
}

const previewRoot = path.resolve('patches/openclaw_skill_manifest_contract_slice_2026-04-03/preview')
const previewSkillsStatus = path.join(previewRoot, 'usr/lib/node_modules/openclaw/dist/skills-status-6dqJ2gft.js')
const previewSkillsCli = path.join(previewRoot, 'usr/lib/node_modules/openclaw/dist/skills-cli-GuSPe1vI.js')
const previewSpotify = path.join(previewRoot, 'usr/lib/node_modules/openclaw/skills/spotify-player/SKILL.md')
const installedSpotify = '/usr/lib/node_modules/openclaw/skills/spotify-player/SKILL.md'

await checkSyntax(previewSkillsStatus)
await checkSyntax(previewSkillsCli)

const currentSpotify = await validateSkillFile(installedSpotify)
const patchedSpotify = await validateSkillFile(previewSpotify)

const currentCodes = issueCodes(currentSpotify)
assert(currentCodes.has('DUPLICATE_INSTALL_ID'), 'expected current spotify-player to expose duplicate install ids')
assert(currentCodes.has('UNKNOWN_INSTALL_KEY'), 'expected current spotify-player to expose unsupported tap key')
assert(patchedSpotify.contract.issues.length === 0, 'expected patched spotify-player preview to be contract-clean')

const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'skill-contract-'))
const badSkillDir = path.join(tmpDir, 'skills', 'bad-contract')
await fs.mkdir(badSkillDir, { recursive: true })
await fs.writeFile(path.join(badSkillDir, 'SKILL.md'), `---
name: bad-contract
description: Synthetic malformed skill for validator coverage.
user-invocable: maybe
disable-model-invocation: yes
metadata:
  {
    "openclaw": {
      "skillKey": "bad.contract",
      "primaryEnv": "OPENAI_API_KEY",
      "requires": {
        "env": ["OTHER_KEY"],
        "oops": ["x"]
      },
      "install": [
        { "id": "dup", "kind": "brew" },
        { "id": "dup", "kind": "magic", "url": "https://example.com/tool.tgz" }
      ],
      "mystery": true
    }
  }
---

# Synthetic malformed skill
`)

const badResult = await validateSkillFile(path.join(badSkillDir, 'SKILL.md'))
const badCodes = issueCodes(badResult)
for (const code of [
  'INVALID_INVOCATION_BOOLEAN',
  'INVALID_SKILL_KEY',
  'PRIMARY_ENV_NOT_DECLARED',
  'UNKNOWN_REQUIRES_KEY',
  'MISSING_INSTALL_FIELD',
  'UNSUPPORTED_INSTALL_KIND',
  'DUPLICATE_INSTALL_ID',
  'UNKNOWN_METADATA_KEY',
]) {
  assert(badCodes.has(code), `expected malformed fixture to surface ${code}`)
}

const summary = {
  previewSyntax: 'ok',
  currentSpotifyIssues: currentSpotify.contract.issues,
  patchedSpotifyIssues: patchedSpotify.contract.issues,
  malformedFixtureIssues: badResult.contract.issues,
}

console.log(JSON.stringify(summary, null, 2))
