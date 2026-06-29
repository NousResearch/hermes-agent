import { describe, expect, it } from 'vitest'

import { summarizeShellCommand } from './summarize-command'

describe('summarizeShellCommand', () => {
  it('strips a leading cd and trailing tail + status echo', () => {
    expect(
      summarizeShellCommand(
        'cd /Users/me/www/bb-rainbows && pnpm run lint 2>&1 | tail -10; echo "lint_exit=${PIPESTATUS[0]}"'
      )
    ).toBe('pnpm run lint')
  })

  it('keeps flags on the surviving command', () => {
    expect(summarizeShellCommand('cd /x && pnpm run preview --port 4317 2>&1')).toBe('pnpm run preview --port 4317')
  })

  it('drops a source/activate prefix', () => {
    expect(summarizeShellCommand('source .venv/bin/activate && pytest -q')).toBe('pytest -q')
  })

  it('skips leading env assignments', () => {
    expect(summarizeShellCommand('cd /x && NODE_ENV=test FOO=bar vitest run 2>&1 | tail -5')).toBe(
      'NODE_ENV=test FOO=bar vitest run'
    )
  })

  it('compacts a genuine multi-command compound without listing every command', () => {
    const compound = 'git add -A && git commit -m "wip"'
    expect(summarizeShellCommand(compound)).toBe('git add -A + 1 command')
  })

  it('leaves a single bare command untouched', () => {
    expect(summarizeShellCommand('git status --short')).toBe('git status --short')
  })

  it('does not split on operators inside quotes', () => {
    const cmd = 'git commit -m "fix: a | b && c"'
    expect(summarizeShellCommand(cmd)).toBe(cmd)
  })

  it('does not strip a redirection-looking char inside quotes', () => {
    expect(summarizeShellCommand('cd /x && git commit -m "a > b"')).toBe('git commit -m "a > b"')
  })

  it('handles empty / whitespace input', () => {
    expect(summarizeShellCommand('')).toBe('')
    expect(summarizeShellCommand('   ')).toBe('')
  })

  it('returns the original when every segment is plumbing', () => {
    const allSetup = 'cd /x && export FOO=1'
    expect(summarizeShellCommand(allSetup)).toBe(allSetup)
  })

  it('collapses 2>&1 redirection on a plain pipeline', () => {
    expect(summarizeShellCommand('cd /x && tsc --noEmit 2>&1 | tail -20')).toBe('tsc --noEmit')
  })

  it('strips a pipe tail from a single bare command (no leading cd)', () => {
    expect(summarizeShellCommand('tsc --noEmit 2>&1 | tail -20')).toBe('tsc --noEmit')
  })

  it('drops a leading echo banner around a single command', () => {
    expect(
      summarizeShellCommand(
        'echo "--- proto pnpm direct ---"; ~/.proto/tools/node/24.11.0/bin/pnpm --version 2>&1 | tail -3'
      )
    ).toBe('~/.proto/tools/node/24.11.0/bin/pnpm --version')
  })

  it('drops echo banners on both sides plus the trailing status echo', () => {
    expect(summarizeShellCommand('echo "--- build ---"; npm run build 2>&1 | tail -5; echo "build_exit=$?"')).toBe(
      'npm run build'
    )
  })

  it('compacts a genuine multi-command probe from session 20260624_231846_bdbd1e', () => {
    const probe = 'which node pnpm corepack; node -v; corepack --version 2>&1'
    expect(summarizeShellCommand(probe)).toBe('which node pnpm corepack + 2 commands')
  })

  it('compacts the corepack diagnostic command from session 20260624_231846_bdbd1e', () => {
    expect(
      summarizeShellCommand(
        'which node pnpm corepack; node -v; echo "---"; corepack --version 2>&1; echo "---pnpm via corepack---"; pnpm --version 2>&1 | tail -5'
      )
    ).toBe('which node pnpm corepack + 3 commands')
  })

  it('compacts the proto/cache probe from session 20260624_231846_bdbd1e', () => {
    expect(
      summarizeShellCommand(
        'echo "--- proto pnpm direct ---"; ~/.proto/tools/node/24.11.0/bin/pnpm --version 2>&1 | tail -3; echo "--- proto node ---"; ls ~/.proto/tools/node/ 2>&1; echo "--- corepack cache ---"; ls ~/.cache/node/corepack/v1/pnpm/ 2>&1'
      )
    ).toBe('~/.proto/tools/node/24.11.0/bin/pnpm --version + 2 commands')
  })

  it('summarizes the successful lint command from session 20260624_231846_bdbd1e', () => {
    expect(
      summarizeShellCommand(
        'cd /Users/brooklyn/www/bb-rainbows && pnpm run lint 2>&1 | tail -20; echo "lint_exit=${PIPESTATUS[0]}"'
      )
    ).toBe('pnpm run lint')
  })

  it('summarizes a background build command from session 20260624_231846_bdbd1e', () => {
    expect(
      summarizeShellCommand(
        'cd /Users/brooklyn/www/bb-rainbows && pnpm run build 2>&1 | tail -20; echo "build_exit=${PIPESTATUS[0]}"'
      )
    ).toBe('pnpm run build')
  })
})
