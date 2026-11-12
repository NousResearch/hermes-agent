const fs = require('node:fs')
const path = require('node:path')
const rules = require('./known-failures.json')

function matchKnownFailure({ platform, phase, commit, installMethod, updateMethod, error, logs }) {
  if (phase !== 'update' || !/^[0-9a-f]{40}$/.test(commit || '')) return null
  return rules.find(rule =>
    (rule.platforms || ['windows']).includes(platform) &&
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

// Each harness keeps its own work-root layout: shas.json + hermes-home on
// windows-e2e.ps1, shas.env + home/.hermes on macos-desktop-e2e.sh.
const WORK_ROOTS = {
  windows: root => ({
    state: JSON.parse(fs.readFileSync(path.join(root, 'shas.json'), 'utf8').replace(/^\uFEFF/, '')),
    logs: {
      update: readOptional(path.join(root, 'logs', 'update.log')),
      desktop: readOptional(path.join(root, 'hermes-home', 'logs', 'desktop.log')),
    },
  }),
  macos: root => {
    const env = Object.fromEntries(fs.readFileSync(path.join(root, 'shas.env'), 'utf8')
      .split(/\r?\n/).filter(line => line.includes('=')).map(line => [line.slice(0, line.indexOf('=')), line.slice(line.indexOf('=') + 1)]))
    return {
      state: { old: env.OLD_SHA, current: env.TARGET_SHA, old_ref: env.OLD_REF },
      logs: { bootstrap: readOptional(path.join(root, 'home', '.hermes', 'logs', 'bootstrap-installer.log')) },
    }
  },
}

function classifyWorkRoot(root, installMethod, updateMethod, error, platform = 'windows') {
  const { state, logs } = WORK_ROOTS[platform](root)
  const rule = matchKnownFailure({
    platform, phase: 'update', commit: state.old, installMethod, updateMethod, error, logs,
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
  const [root, install, update, error, platform] = process.argv.slice(2)
  try {
    const receipt = classifyWorkRoot(root, install, update, error, platform || 'windows')
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
