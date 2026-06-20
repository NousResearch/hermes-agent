const fsp = require('fs/promises');
const path = require('path');

function summarizePatch(diffText) {
  const files = [];
  let current;
  const finish = () => {
    if (current) files.push(current);
    current = undefined;
  };
  for (const line of String(diffText || '').replace(/\r\n/g, '\n').split('\n')) {
    if (line.startsWith('diff --git ')) {
      finish();
      const parts = line.split(/\s+/);
      current = { path: stripDiffPath(parts[3] || parts[2]), additions: 0, deletions: 0, hunks: 0 };
      continue;
    }
    if (line.startsWith('--- ') && !current) {
      current = { path: stripDiffPath(line.slice(4)), additions: 0, deletions: 0, hunks: 0 };
      continue;
    }
    if (line.startsWith('+++ ') && current) {
      const newPath = stripDiffPath(line.slice(4));
      if (newPath !== '/dev/null') current.path = newPath;
      continue;
    }
    if (!current) continue;
    if (line.startsWith('@@ ')) current.hunks += 1;
    else if (line.startsWith('+') && !line.startsWith('+++ ')) current.additions += 1;
    else if (line.startsWith('-') && !line.startsWith('--- ')) current.deletions += 1;
  }
  finish();
  const cleaned = files.filter((file) => file.path && (file.additions || file.deletions || file.hunks));
  return {
    files: cleaned,
    additions: cleaned.reduce((sum, file) => sum + file.additions, 0),
    deletions: cleaned.reduce((sum, file) => sum + file.deletions, 0),
    hunks: cleaned.reduce((sum, file) => sum + file.hunks, 0)
  };
}

function stripDiffPath(value) {
  if (!value) return value;
  const cleaned = value.trim().split('\t')[0].split(' ')[0];
  if (cleaned === '/dev/null') return cleaned;
  return cleaned.startsWith('a/') || cleaned.startsWith('b/') ? cleaned.slice(2) : cleaned;
}

async function detectTestCommand(cwd, fsApi = fsp) {
  const exists = async (name) => {
    try { await fsApi.stat(path.join(cwd, name)); return true; } catch { return false; }
  };
  const readJson = async (name) => {
    try { return JSON.parse(await fsApi.readFile(path.join(cwd, name), 'utf8')); } catch { return undefined; }
  };

  const packageJson = await readJson('package.json');
  if (packageJson) {
    const scripts = packageJson.scripts || {};
    const manager = await detectPackageManager(cwd, exists);
    const preferred = pickPackageScript(scripts, packageJson.name);
    if (preferred) return packageScriptCommand(manager, preferred);
    if (manager === 'bun') return 'bun test';
    if (manager === 'pnpm') return 'pnpm test';
    if (manager === 'yarn') return 'yarn test';
    return 'npm test';
  }

  const vscodeExtensionPackage = await readJson(path.join('apps', 'vscode', 'package.json'));
  if (vscodeExtensionPackage?.scripts?.['package:check']) {
    return 'npm run package:check --workspace apps/vscode';
  }

  if (await exists('pyproject.toml') || await exists('pytest.ini')) {
    if (await exists('uv.lock')) return 'uv run pytest';
    return 'python -m pytest';
  }
  if (await exists('Package.swift')) return 'swift test';
  if (await exists('go.mod')) return 'go test ./...';
  if (await exists('Cargo.toml')) return 'cargo test';
  if (await exists('Makefile')) return 'make test';
  return 'echo "No standard test command detected. Edit this command and press Enter."';
}

async function detectPackageManager(_cwd, exists) {
  if (await exists('pnpm-lock.yaml')) return 'pnpm';
  if (await exists('yarn.lock')) return 'yarn';
  if (await exists('bun.lockb') || await exists('bun.lock')) return 'bun';
  return 'npm';
}

function pickPackageScript(scripts, packageName = '') {
  if (packageName === 'hermes-vscode' && scripts['package:check']) return 'package:check';
  if (scripts.test) return 'test';
  if (scripts['package:check']) return 'package:check';
  if (scripts.check) return 'check';
  if (scripts.lint) return 'lint';
  return undefined;
}

function packageScriptCommand(manager, script) {
  if (manager === 'npm') return script === 'test' ? 'npm test' : `npm run ${script}`;
  if (manager === 'yarn') return script === 'test' ? 'yarn test' : `yarn ${script}`;
  if (manager === 'pnpm') return script === 'test' ? 'pnpm test' : `pnpm run ${script}`;
  if (manager === 'bun') return script === 'test' ? 'bun test' : `bun run ${script}`;
  return `npm run ${script}`;
}

function buildHermesPrompt(userText, contextBlock, options = {}) {
  const mode = options.mode || 'ask';
  const patchInstruction = options.wantsPatch || isPatchMode(mode)
    ? '\nIf code edits are useful, return them as a unified diff in a ```diff fenced block. Keep explanations short and include verification commands.'
    : '\nWhen code edits are useful, prefer a short explanation plus a unified diff in a ```diff fenced block.';
  return `You are Hermes running inside a VS Code/Cursor extension. Use the supplied editor context. Be concise, operational, and careful with unsaved buffers. Do not expose secrets.\n${modeInstructions(mode)}${patchInstruction}\n\n${contextBlock}\n\n## User request\n\n${userText}`;
}

function buildTerminalDebugPrompt(command, output, status, maxChars = 30000) {
  return `The terminal command below ${status}. Analyze the captured output, identify likely root cause, and recommend the smallest next fix. If a code change is needed, return a unified diff.\n\nCommand:\n\n\`\`\`sh\n${command}\n\`\`\`\n\nOutput:\n\n\`\`\`text\n${truncateText(output || '(no output)', maxChars)}\n\`\`\``;
}

function buildTestAnalysisPrompt(command, output, status, maxChars = 30000) {
  return `The detected test command \`${command}\` ${status}. Analyze this output, identify likely root cause, and recommend the smallest next fix. If a code change is needed, return a unified diff.\n\n\`\`\`text\n${truncateText(output || '(no output)', maxChars)}\n\`\`\``;
}

function isPatchMode(mode) {
  return ['edit', 'debug', 'refactor', 'test', 'security'].includes(mode);
}

function modeInstructions(mode) {
  const modes = {
    ask: 'Mode: general coding assistant. Answer directly and cite relevant files/context.',
    explain: 'Mode: explain. Clarify what the selected code/file does, important data flow, risks, and gotchas.',
    edit: 'Mode: edit. Produce a focused patch that preserves behavior unless the user asks otherwise.',
    review: 'Mode: review. Summarize risk, likely breakage, missing tests, and concrete recommended changes.',
    test: 'Mode: test. Identify or add targeted tests and suggest the smallest verification command.',
    debug: 'Mode: debug. Form a hypothesis, inspect evidence from context, propose fixes, and include verification steps.',
    refactor: 'Mode: refactor. Improve structure/readability while preserving behavior and minimizing churn.',
    security: 'Mode: security review. Focus on trust boundaries, secrets, authz/authn, injection, data exposure, and safe mitigations.',
    commit: 'Mode: commit message. Summarize the current change as a concise conventional commit with optional body.'
  };
  return modes[mode] || modes.ask;
}

function truncateText(text, max) {
  const value = text || '';
  if (!max || value.length <= max) return value;
  return value.slice(0, max) + `\n\n...[truncated ${value.length - max} chars]`;
}

module.exports = {
  summarizePatch,
  stripDiffPath,
  detectTestCommand,
  pickPackageScript,
  packageScriptCommand,
  buildHermesPrompt,
  buildTerminalDebugPrompt,
  buildTestAnalysisPrompt,
  modeInstructions,
  truncateText
};
