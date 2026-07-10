// Git-driven worktree operations for the desktop "Start work" flow: spin up a
// fresh worktree the lightest way (`git worktree add -b`), list real worktrees,
// and remove them. Git is the source of truth; the renderer just drives these.

import { execFile } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { resolveRequestedPathForIpc } from './hardening.ts'

function runGit(gitBin, args, cwd): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      gitBin,
      args,
      { cwd, windowsHide: true, timeout: 30_000, maxBuffer: 8 * 1024 * 1024 },
      (err, stdout, stderr) => {
        if (err) {
          err.stderr = String(stderr || '')
          reject(err)

          return
        }

        resolve(String(stdout || ''))
      }
    )
  })
}

// Parse `git worktree list --porcelain`. The first record is the main worktree.
function parseWorktrees(out) {
  const trees = []
  let cur = null

  for (const line of out.split('\n')) {
    if (line.startsWith('worktree ')) {
      if (cur) {
        trees.push(cur)
      }

      cur = { path: line.slice(9).trim(), branch: null, detached: false, bare: false, locked: false }
    } else if (!cur) {
      continue
    } else if (line.startsWith('branch ')) {
      cur.branch = line
        .slice(7)
        .trim()
        .replace(/^refs\/heads\//, '')
    } else if (line === 'detached') {
      cur.detached = true
    } else if (line === 'bare') {
      cur.bare = true
    } else if (line.startsWith('locked')) {
      cur.locked = true
    }
  }

  if (cur) {
    trees.push(cur)
  }

  return trees
}

async function listWorktrees(repoPath, gitBin) {
  let resolved

  try {
    resolved = resolveRequestedPathForIpc(repoPath, { purpose: 'Worktree list' })
  } catch {
    return []
  }

  try {
    const out = await runGit(gitBin, ['worktree', 'list', '--porcelain'], resolved)

    return parseWorktrees(out).map((tree, index) => ({
      path: tree.path,
      branch: tree.branch,
      isMain: index === 0,
      detached: tree.detached,
      locked: tree.locked
    }))
  } catch {
    return []
  }
}

// A git-ref-safe branch name (spaces → "-", drop forbidden chars, trim edges),
// or "" when nothing usable remains. Mirrors the renderer's `gitRef`, so a bad
// value can't reach `git` no matter the caller (the GUI also enforces live).
function sanitizeBranch(name) {
  return String(name || '')
    .replace(/\s+/g, '-')
    .replace(/[^\w./-]/g, '')
    .replace(/-{2,}/g, '-')
    .replace(/\/{2,}/g, '/')
    .replace(/\.{2,}/g, '.')
    .replace(/^[-./]+|[-./]+$/g, '')
}

function slugify(name) {
  const slug = String(name || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40)
    .replace(/-+$/g, '')

  return slug || 'work'
}

const TRUNK_BRANCHES = ['main', 'master']

async function gitLine(gitBin, args, cwd) {
  try {
    return (await runGit(gitBin, args, cwd)).trim()
  } catch {
    return ''
  }
}

async function defaultBranch(gitBin, cwd) {
  const remote = (
    await gitLine(gitBin, ['symbolic-ref', '--quiet', '--short', 'refs/remotes/origin/HEAD'], cwd)
  ).replace(/^origin\//, '')

  if (remote) {
    return remote
  }

  const configured = await gitLine(gitBin, ['config', '--get', 'init.defaultBranch'], cwd)

  if (configured) {
    return configured
  }

  for (const branch of TRUNK_BRANCHES) {
    if (await gitLine(gitBin, ['show-ref', '--verify', `refs/heads/${branch}`], cwd)) {
      return branch
    }
  }

  return ''
}

// A brand-new project folder isn't a git repo — and a freshly-init'd one has no
// commit to branch from — so `git worktree add` would fail. Make the dir a repo
// with a root commit on the user's behalf so worktrees "just work". No-op for a
// repo that already has commits; never touches the user's files (the seed commit
// is `--allow-empty`), and never inits a dir that already lives inside a repo.
async function ensureGitRepo(gitBin, dir) {
  let needsRoot = false

  try {
    const inside = (await runGit(gitBin, ['rev-parse', '--is-inside-work-tree'], dir)).trim()

    if (inside !== 'true') {
      await runGit(gitBin, ['init'], dir)
      needsRoot = true
    } else {
      // Repo exists; a worktree still needs a HEAD to branch from.
      try {
        await runGit(gitBin, ['rev-parse', '--verify', 'HEAD'], dir)
      } catch {
        needsRoot = true
      }
    }
  } catch {
    await runGit(gitBin, ['init'], dir)
    needsRoot = true
  }

  if (needsRoot) {
    // Inline identity so the seed commit lands even with no global git config.
    await runGit(
      gitBin,
      [
        '-c',
        'user.email=hermes@localhost',
        '-c',
        'user.name=Hermes',
        'commit',
        '--allow-empty',
        '-m',
        'Initial commit'
      ],
      dir
    )
  }
}

// Resolve the repo's MAIN worktree root, so `.worktrees/` always nests under the
// primary checkout even when called from a linked worktree.
async function mainRoot(gitBin, cwd) {
  const list = await listWorktrees(cwd, gitBin)
  const main = list.find(tree => tree.isMain)

  return main ? main.path : cwd
}

function uniqueDir(base) {
  let dir = base
  let n = 1

  while (fs.existsSync(dir)) {
    n += 1
    dir = `${base}-${n}`
  }

  return dir
}

async function addExistingBranchWorktree(gitBin, root, name) {
  const branch = sanitizeBranch(name)

  if (!branch) {
    throw new Error('Branch name is required.')
  }

  if (branch === (await defaultBranch(gitBin, root))) {
    await runGit(gitBin, ['switch', branch], root)

    return { path: root, branch, repoRoot: root }
  }

  const dir = uniqueDir(path.join(root, '.worktrees', slugify(branch)))
  await runGit(gitBin, ['worktree', 'add', dir, branch], root)

  return { path: dir, branch, repoRoot: root }
}

async function addWorktree(repoPath, options, gitBin) {
  const resolved = resolveRequestedPathForIpc(repoPath, { purpose: 'Worktree add' })
  // A new project's folder may not be a git repo yet — init it (with a root
  // commit) so the worktree has something to branch from.
  await ensureGitRepo(gitBin, resolved)
  const root = await mainRoot(gitBin, resolved)
  const opts = options || {}

  if (opts.existingBranch) {
    return addExistingBranchWorktree(gitBin, root, opts.existingBranch)
  }

  const slug = slugify(opts.name || `work-${Date.now().toString(36)}`)
  const branch = sanitizeBranch(opts.branch) || `hermes/${slug}`
  const dir = uniqueDir(path.join(root, '.worktrees', slug))

  const args = ['worktree', 'add', '-b', branch, dir]

  if (opts.base) {
    args.push(String(opts.base))
  }

  try {
    await runGit(gitBin, args, root)
  } catch (err) {
    // Branch name may already exist — retry checking out the existing branch
    // into a fresh worktree dir instead of failing the whole flow.
    if (/already exists/i.test(err.stderr || '')) {
      await runGit(gitBin, ['worktree', 'add', dir, branch], root)
    } else {
      throw err
    }
  }

  return { path: dir, branch, repoRoot: root }
}

async function removeWorktree(repoPath, worktreePath, options, gitBin) {
  const resolvedRepo = resolveRequestedPathForIpc(repoPath, { purpose: 'Worktree remove (repo)' })
  const resolvedTree = resolveRequestedPathForIpc(worktreePath, { purpose: 'Worktree remove (tree)' })
  const root = await mainRoot(gitBin, resolvedRepo)
  const args = ['worktree', 'remove']

  if (options && options.force) {
    args.push('--force')
  }

  args.push(resolvedTree)
  await runGit(gitBin, args, root)

  return { removed: resolvedTree }
}

// List local branches for the "convert a branch into a worktree" picker, most
// recently committed first. Each carries whether it's already checked out in a
// worktree and, when checked out, that worktree's path. Empty on a non-repo /
// remote backend where the probe can't run.
async function listBranches(repoPath, gitBin) {
  let resolved

  try {
    resolved = resolveRequestedPathForIpc(repoPath, { purpose: 'Branch list' })
  } catch {
    return []
  }

  try {
    const out = await runGit(
      gitBin,
      ['for-each-ref', '--format=%(refname:short)', '--sort=-committerdate', 'refs/heads'],
      resolved
    )

    const trees = await listWorktrees(resolved, gitBin)
    const pathByBranch = new Map(trees.filter(tree => tree.branch).map(tree => [tree.branch, tree.path]))
    const trunk = await defaultBranch(gitBin, resolved)

    return out
      .split('\n')
      .map(line => line.trim())
      .filter(Boolean)
      .map(name => ({
        name,
        checkedOut: pathByBranch.has(name),
        isDefault: Boolean(trunk && name === trunk),
        worktreePath: pathByBranch.get(name) || null
      }))
  } catch {
    return []
  }
}

async function switchBranch(repoPath, branch, gitBin) {
  const resolved = resolveRequestedPathForIpc(repoPath, { purpose: 'Branch switch' })
  const target = sanitizeBranch(branch)

  if (!target) {
    throw new Error('Branch name is required.')
  }

  await runGit(gitBin, ['switch', target], resolved)

  return { branch: target }
}

export {
  addWorktree,
  ensureGitRepo,
  listBranches,
  listWorktrees,
  parseWorktrees,
  removeWorktree,
  sanitizeBranch,
  switchBranch
}
