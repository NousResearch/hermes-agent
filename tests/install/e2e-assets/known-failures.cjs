const fs = require('node:fs')
const path = require('node:path')
const rules = require('./known-failures.json')

function matchKnownFailure({ platform, phase, commit, installMethod, updateMethod, error, logs }) {
  if (platform !== 'windows' || phase !== 'update' || !/^[0-9a-f]{40}$/.test(commit || '')) return null
  return rules.find(rule =>
    rule.commits.includes(commit) &&
    rule.cases.some(([install, update]) => install === installMethod && update === updateMethod) &&
    rule.errors.some(pattern => new RegExp(pattern).test(error || '')) &&
    rule.signatures.every(pattern => new RegExp(pattern, 'i').test(logs[rule.log] || '')),
  ) || null
}

function readOptional(file) {
  try { return fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, '') } catch (error) {
    if (error.code === 'ENOENT') return ''
    throw error
  }
}

function classifyWorkRoot(root, installMethod, updateMethod, error) {
  const state = JSON.parse(fs.readFileSync(path.join(root, 'shas.json'), 'utf8').replace(/^\uFEFF/, ''))
  const rule = matchKnownFailure({
    platform: 'windows', phase: 'update', commit: state.old, installMethod, updateMethod, error,
    logs: {
      update: readOptional(path.join(root, 'logs', 'update.log')),
      desktop: readOptional(path.join(root, 'hermes-home', 'logs', 'desktop.log')),
    },
  })
  if (!rule) return null
  return {
    id: rule.id, title: rule.title, explanation: rule.explanation, evidence: rule.evidence,
    commit: state.old, target: state.current, installRef: state.old_ref,
    installMethod, updateMethod, error,
  }
}

module.exports = { matchKnownFailure, classifyWorkRoot, rules }

if (require.main === module) {
  const [root, install, update, error] = process.argv.slice(2)
  try {
    const receipt = classifyWorkRoot(root, install, update, error)
    if (!receipt) process.exitCode = 1
    else {
      fs.writeFileSync(path.join(root, 'known-failure.json'), JSON.stringify(receipt, null, 2) + '\n')
      console.log(JSON.stringify(receipt))
    }
  } catch (error) {
    console.error(`known-failure classification failed: ${error.message}`)
    process.exitCode = 2
  }
}
