/**
 * Pure helpers for choosing a remote URL during passive update checks.
 *
 * A public install can use an SSH or HTTPS origin. Passive checks against the
 * official repo substitute the public HTTPS `ls-remote` path, which needs no
 * auth and cannot prompt. This also avoids unexplained hardware-touch prompts
 * for users with FIDO2/passkey-backed SSH keys. Active update/apply flows are
 * left unchanged.
 *
 * Extracted from main.ts so the security-critical remote detection is unit
 * testable without booting Electron (main.ts requires('electron') at load).
 */

const OFFICIAL_REPO_HTTPS_URL = 'https://github.com/NousResearch/hermes-agent.git'
const OFFICIAL_REPO_CANONICAL = 'github.com/nousresearch/hermes-agent'

// Normalize common GitHub remote URL forms to `host/owner/repo` (lowercased,
// no trailing slash, no .git suffix) so SSH and HTTPS forms of the same repo
// compare equal.
function canonicalGitHubRemote(url) {
  if (!url) {
    return ''
  }

  let value = String(url).trim()

  if (value.startsWith('git@github.com:')) {
    value = `github.com/${value.slice('git@github.com:'.length)}`
  } else if (value.startsWith('ssh://git@github.com/')) {
    value = `github.com/${value.slice('ssh://git@github.com/'.length)}`
  } else {
    try {
      const parsed = new URL(value)

      if (parsed.hostname && parsed.pathname) {
        value = `${parsed.hostname}${parsed.pathname}`
      }
    } catch {
      // Leave non-URL forms unchanged.
    }
  }

  value = value.trim().replace(/\/+$/, '')

  if (value.endsWith('.git')) {
    value = value.slice(0, -4)
  }

  return value.toLowerCase()
}

function isSshRemote(url) {
  const value = String(url || '')
    .trim()
    .toLowerCase()

  return value.startsWith('git@') || value.startsWith('ssh://')
}

function isOfficialSshRemote(url) {
  return isSshRemote(url) && canonicalGitHubRemote(url) === OFFICIAL_REPO_CANONICAL
}

function isOfficialRemote(url) {
  return canonicalGitHubRemote(url) === OFFICIAL_REPO_CANONICAL
}

function nonInteractiveGitEnv(env = {}) {
  return { ...env, GIT_TERMINAL_PROMPT: '0' }
}

export {
  canonicalGitHubRemote,
  isOfficialRemote,
  isOfficialSshRemote,
  isSshRemote,
  nonInteractiveGitEnv,
  OFFICIAL_REPO_CANONICAL,
  OFFICIAL_REPO_HTTPS_URL
}
