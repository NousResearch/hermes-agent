/**
 * Parse --profile <name> from command-line arguments.
 *
 * When the desktop app is launched with `--profile <name>`, this module
 * extracts and validates the profile name so it can be persisted before
 * the backend starts.
 *
 * @param {string[]} argv - Command-line arguments (defaults to process.argv)
 * @returns {string|null} Validated profile name, or null if not present/invalid
 */
function parseCliProfile(argv = process.argv) {
  const args = argv || []
  const idx = args.indexOf('--profile')
  if (idx === -1 || idx + 1 >= args.length) return null

  const name = String(args[idx + 1] || '').trim()
  if (!name) return null

  // Mirror PROFILE_NAME_RE from main.cjs: lowercase alphanumeric, hyphens,
  // underscores, 1-64 chars.  "default" is always valid.
  if (name !== 'default' && !/^[a-z0-9][a-z0-9_-]{0,63}$/.test(name)) {
    return null
  }

  return name
}

module.exports = { parseCliProfile }
