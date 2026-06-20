const vscode = require('vscode');
const cp = require('child_process');
const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const crypto = require('crypto');
const { extractUnifiedDiff, parsePatch, applyPatchToContent, safeJoin } = require('./patchParser');
const { AcpClient } = require('./acpClient');

let provider;

function activate(context) {
  const output = vscode.window.createOutputChannel('Hermes Code');
  provider = new HermesChatProvider(context, output);
  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 40);
  statusBar.text = '$(sparkle) Hermes';
  statusBar.tooltip = 'Open Hermes Code';
  statusBar.command = 'hermesCode.openChat';
  statusBar.show();

  context.subscriptions.push(output, statusBar, provider);
  context.subscriptions.push(vscode.window.registerWebviewViewProvider('hermesCode.chatView', provider, { webviewOptions: { retainContextWhenHidden: true } }));
  context.subscriptions.push(vscode.commands.registerCommand('hermesCode.openChat', () => vscode.commands.executeCommand('hermesCode.chatView.focus')));
  context.subscriptions.push(vscode.commands.registerCommand('hermesCode.askSelection', () => provider.askSelection()));
  context.subscriptions.push(vscode.commands.registerCommand('hermesCode.editSelection', () => provider.editSelection()));
  context.subscriptions.push(vscode.commands.registerCommand('hermesCode.reviewDiff', () => provider.reviewDiff()));
  context.subscriptions.push(vscode.commands.registerCommand('hermesCode.explainDiagnostics', () => provider.explainDiagnostics()));
  context.subscriptions.push(vscode.commands.registerCommand('hermesCode.runCommandWithContext', () => provider.runCommandWithContext()));
  context.subscriptions.push(vscode.commands.registerCommand('hermesCode.previewPatch', () => provider.previewLastPatch()));
  context.subscriptions.push(vscode.commands.registerCommand('hermesCode.applyPatch', () => provider.applyLastPatch()));
  context.subscriptions.push(vscode.commands.registerCommand('hermesCode.copyLastPatch', () => provider.copyLastPatch()));
  context.subscriptions.push(vscode.commands.registerCommand('hermesCode.clearChat', () => provider.clearChat()));
  context.subscriptions.push(vscode.commands.registerCommand('hermesCode.configureHermesPath', () => provider.configureHermesPath()));
  context.subscriptions.push(vscode.languages.registerCodeActionsProvider({ scheme: 'file' }, new HermesCodeActionProvider(), { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] }));
}

function deactivate() {}

class HermesCodeActionProvider {
  provideCodeActions(_document, _range, context) {
    if (!context.diagnostics || !context.diagnostics.length) return [];
    const explain = new vscode.CodeAction('Explain diagnostics with Hermes', vscode.CodeActionKind.QuickFix);
    explain.command = { command: 'hermesCode.explainDiagnostics', title: 'Explain diagnostics with Hermes' };
    const fix = new vscode.CodeAction('Fix diagnostics with Hermes', vscode.CodeActionKind.QuickFix);
    fix.command = { command: 'hermesCode.explainDiagnostics', title: 'Fix diagnostics with Hermes' };
    return [explain, fix];
  }
}

class HermesChatProvider {
  constructor(context, output) {
    this.context = context;
    this.output = output;
    this.view = undefined;
    this.running = false;
    this.lastPatchPath = path.join(context.globalStorageUri.fsPath, 'last-hermes.patch');
    this.historyKey = 'hermesCode.chatHistory';
    this.acpClient = undefined;
  }

  dispose() {
    if (this.child) this.child.kill('SIGTERM');
    if (this.acpClient) this.acpClient.dispose();
  }

  resolveWebviewView(view) {
    this.view = view;
    view.webview.options = { enableScripts: true };
    view.webview.html = this.html();
    view.webview.onDidReceiveMessage(async (message) => {
      try {
        if (message.type === 'ready') this.restoreHistory();
        if (message.type === 'ask') await this.ask(message.text || '', message.options || {});
        if (message.type === 'previewPatch') await this.previewLastPatch();
        if (message.type === 'applyPatch') await this.applyLastPatch();
        if (message.type === 'copyPatch') await this.copyLastPatch();
        if (message.type === 'discardPatch') await this.discardLastPatch();
        if (message.type === 'revertPatch') await this.revertLastPatch();
        if (message.type === 'runTests') await this.runDetectedTests();
        if (message.type === 'runTestsAndAnalyze') await this.runTestsAndAnalyze();
        if (message.type === 'inspectContext') await this.inspectContext(message.options || {});
        if (message.type === 'reviewDiff') await this.reviewDiff();
        if (message.type === 'diagnostics') await this.explainDiagnostics();
        if (message.type === 'terminal') await this.runCommandWithContext();
        if (message.type === 'clear') await this.clearChat();
        if (message.type === 'configure') await this.configureHermesPath();
        if (message.type === 'stop') this.stopCurrentRun();
      } catch (err) {
        this.output.appendLine(String(err && err.stack ? err.stack : err));
        this.post('error', String(err && err.message ? err.message : err));
      }
    });
  }

  post(type, value) {
    this.view?.webview.postMessage({ type, value });
  }

  persist(role, text) {
    const history = this.context.workspaceState.get(this.historyKey, []);
    history.push({ role, text, at: new Date().toISOString() });
    this.context.workspaceState.update(this.historyKey, history.slice(-60));
  }

  restoreHistory() {
    const history = this.context.workspaceState.get(this.historyKey, []);
    this.post('restore', history);
    const patchExists = fs.existsSync(this.lastPatchPath);
    this.post('patchState', patchExists ? 'Patch available' : 'No patch yet');
    if (patchExists) {
      fsp.readFile(this.lastPatchPath, 'utf8')
        .then((patchText) => this.post('patchDetails', summarizePatch(patchText)))
        .catch((error) => this.output.appendLine(`Patch summary skipped: ${error.message}`));
    } else {
      this.post('patchDetails', null);
    }
  }

  async clearChat() {
    await this.context.workspaceState.update(this.historyKey, []);
    this.post('restore', []);
  }

  async configureHermesPath() {
    const config = vscode.workspace.getConfiguration('hermesCode');
    const current = config.get('hermesCommand', 'hermes');
    const value = await vscode.window.showInputBox({ prompt: 'Hermes command or absolute path', value: current });
    if (value) await config.update('hermesCommand', value, vscode.ConfigurationTarget.Global);
  }

  async ask(text, options = {}) {
    if (!text.trim()) return;
    const workspace = getWorkspaceFolder();
    const contextBlock = await collectEditorContext(options, this.output);
    const prompt = buildPrompt(text, contextBlock, options);
    await this.runHermes(prompt, workspace?.uri.fsPath || process.cwd(), text);
  }

  async askSelection() {
    const editor = vscode.window.activeTextEditor;
    const selected = getSelectionText(editor);
    const defaultQuestion = selected ? 'Explain this selection and call out important risks or improvements.' : 'Explain the active file.';
    const question = await vscode.window.showInputBox({ prompt: 'Ask Hermes about the current selection/file', value: defaultQuestion });
    if (question) await this.ask(question, { forceSelection: true, includeSelection: true, includeFile: !selected });
  }

  async editSelection() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.selection.isEmpty) {
      vscode.window.showWarningMessage('Select code before asking Hermes to edit it.');
      return;
    }
    const instruction = await vscode.window.showInputBox({ prompt: 'How should Hermes edit the selection?', value: 'Improve this code while preserving behavior. Return a unified diff only.' });
    if (!instruction) return;
    await this.ask(instruction, { forceSelection: true, includeSelection: true, includeFile: false, includeGitDiff: false, wantsPatch: true });
  }

  async reviewDiff() {
    const workspace = getWorkspaceFolder();
    if (!workspace) {
      vscode.window.showWarningMessage('Open a workspace before reviewing a diff.');
      return;
    }
    const diff = await getGitDiff(workspace.uri.fsPath);
    if (!diff.trim()) {
      vscode.window.showInformationMessage('No Git diff found in this workspace.');
      return;
    }
    await this.ask('Review this Git diff. Summarize risk, likely breakage, missing tests, and concrete recommended changes. Do not apply edits unless asked.', { includeFile: false, includeDiagnostics: true, includeInstructions: true, extraContext: `## Current Git diff\n\n\`\`\`diff\n${truncate(diff, getMaxContextChars())}\n\`\`\`` });
  }

  async explainDiagnostics() {
    const diagnostics = diagnosticsForActiveEditor();
    if (!diagnostics.trim()) {
      vscode.window.showInformationMessage('No diagnostics found for the active editor.');
      return;
    }
    await this.ask('Explain these diagnostics and recommend fixes. If code changes are needed, return a unified diff.', { includeFile: true, includeDiagnostics: true, includeInstructions: true, extraContext: diagnostics, wantsPatch: true });
  }

  async runCommandWithContext() {
    const command = await vscode.window.showInputBox({ prompt: 'Command to run in the VS Code terminal' });
    if (!command) return;
    const terminal = vscode.window.createTerminal({ name: 'Hermes Code', cwd: getWorkspaceFolder()?.uri.fsPath });
    terminal.show();
    terminal.sendText(`# Hermes Code requested command in ${getWorkspaceFolder()?.uri.fsPath || process.cwd()}`);
    terminal.sendText(command);
  }

  async inspectContext(options = {}) {
    const summary = await collectContextSummary(options);
    this.post('contextSummary', summary);
  }

  stopCurrentRun() {
    if (this.acpClient) {
      this.acpClient.cancel().catch((error) => this.output.appendLine(`ACP cancel failed: ${error.message}`));
      this.post('status', 'Stopping Hermes ACP...');
    }
    if (this.child) {
      this.child.kill('SIGTERM');
      this.post('status', 'Stopping Hermes...');
      this.output.appendLine('Sent SIGTERM to Hermes subprocess.');
    }
  }

  async handleAcpPermission(params) {
    const options = params.options || [];
    if (!options.length) return { outcome: { outcome: 'cancelled' } };
    const labels = options.map((option) => option.name || option.title || option.optionId || option.id || 'Option');
    const picked = await vscode.window.showWarningMessage(
      `Hermes requests permission for ${params.toolCall?.title || 'a tool call'}`,
      { modal: true, detail: JSON.stringify(params.toolCall || {}, null, 2).slice(0, 1200) },
      ...labels
    );
    if (!picked) return { outcome: { outcome: 'cancelled' } };
    const option = options[labels.indexOf(picked)];
    return { outcome: { outcome: 'selected', optionId: option.optionId || option.id } };
  }

  buildHermesArgs(prompt, cwd) {
    const config = vscode.workspace.getConfiguration('hermesCode');
    const baseArgs = [...config.get('hermesArgs', ['chat', '--quiet', '--source', 'vscode'])];
    const args = baseArgs.filter((arg) => arg !== '-q' && arg !== '--query');
    const sessionMode = config.get('sessionMode', 'workspace');
    if (sessionMode !== 'fresh') {
      args.push('--continue', config.get('sessionName', '') || workspaceSessionName(cwd));
    }
    args.push('-q', prompt);
    return args;
  }

  async runHermes(prompt, cwd, displayText) {
    const backend = vscode.workspace.getConfiguration('hermesCode').get('backend', 'acp');
    if (backend === 'acp') return this.runHermesAcp(prompt, cwd, displayText);
    return this.runHermesSubprocess(prompt, cwd, displayText);
  }

  async runHermesAcp(prompt, cwd, displayText) {
    if (this.running) {
      vscode.window.showWarningMessage('Hermes is already running. Stop it before starting another request.');
      return;
    }
    await fsp.mkdir(this.context.globalStorageUri.fsPath, { recursive: true });
    const config = vscode.workspace.getConfiguration('hermesCode');
    const command = config.get('hermesCommand', 'hermes');
    const args = config.get('acpArgs', ['acp']);
    if (!this.acpClient) {
      this.acpClient = new AcpClient({ command, args, cwd, env: process.env });
      this.acpClient.permissionHandler = (params) => this.handleAcpPermission(params);
      this.acpClient.on('stderr', (text) => this.output.append(text));
      this.acpClient.on('status', (text) => this.post('status', text));
      this.acpClient.on('partial', (text) => this.post('partial', text));
      this.acpClient.on('tool', (tool) => this.post('tool', `${tool.status || 'tool'}: ${tool.title || tool.toolCallId || 'Hermes tool'}`));
      this.acpClient.on('permission', (params) => this.output.appendLine(`ACP permission requested: ${JSON.stringify(params).slice(0, 1000)}`));
    }
    this.running = true;
    this.post('status', `Running ${command} ${args.join(' ')}...`);
    this.post('user', displayText || promptSummary(prompt));
    this.persist('user', displayText || promptSummary(prompt));
    this.output.appendLine(`\n[${new Date().toISOString()}] ACP ${command} ${args.join(' ')} <prompt:${prompt.length} chars>`);
    try {
      const result = await this.acpClient.prompt(prompt, cwd);
      const final = (result.text || '').trim() || '(Hermes returned no output)';
      this.post('assistant', final);
      this.persist('assistant', final);
      const patch = extractUnifiedDiff(final);
      if (patch) {
        await fsp.writeFile(this.lastPatchPath, patch, 'utf8');
        const summary = summarizePatch(patch);
        const count = summary.files.length;
        this.post('patch', `Saved patch with ${count} file(s), +${summary.additions}/-${summary.deletions}.`);
        this.post('patchState', `Patch available (${count} file${count === 1 ? '' : 's'}, +${summary.additions}/-${summary.deletions})`);
        this.post('patchDetails', summary);
      }
      this.post('status', `Ready (${result.response?.stopReason || 'end_turn'})`);
    } catch (error) {
      const msg = `Hermes ACP failed: ${error.message || String(error)}`;
      this.post('error', msg);
      this.persist('error', msg);
      this.output.appendLine(error.stack || msg);
    } finally {
      this.running = false;
    }
  }

  async runHermesSubprocess(prompt, cwd, displayText) {
    if (this.running) {
      vscode.window.showWarningMessage('Hermes is already running. Stop it before starting another request.');
      return;
    }
    await fsp.mkdir(this.context.globalStorageUri.fsPath, { recursive: true });
    const config = vscode.workspace.getConfiguration('hermesCode');
    const command = config.get('hermesCommand', 'hermes');
    const args = this.buildHermesArgs(prompt, cwd);
    this.running = true;
    this.post('status', `Running ${command} ${args.slice(0, -1).join(' ')}...`);
    this.post('user', displayText || promptSummary(prompt));
    this.persist('user', displayText || promptSummary(prompt));
    this.output.appendLine(`\n[${new Date().toISOString()}] ${command} ${args.slice(0, -1).join(' ')} <prompt:${prompt.length} chars>`);

    let stdout = '';
    let stderr = '';
    this.child = cp.spawn(command, args, { cwd, env: process.env });
    this.child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
      this.post('partial', stdout);
    });
    this.child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
      this.output.append(chunk.toString());
    });
    this.child.on('error', (err) => {
      this.running = false;
      this.post('error', `Failed to start Hermes: ${err.message}`);
      this.output.appendLine(`Failed to start Hermes: ${err.stack || err.message}`);
    });
    this.child.on('close', async (code) => {
      this.running = false;
      this.child = undefined;
      if (code !== 0) {
        const msg = `Hermes exited with ${code}.\n\n${stderr || stdout}`;
        this.post('error', msg);
        this.persist('error', msg);
        return;
      }
      const final = stdout.trim() || '(Hermes returned no output)';
      this.post('assistant', final);
      this.persist('assistant', final);
      const patch = extractUnifiedDiff(stdout);
      if (patch) {
        await fsp.writeFile(this.lastPatchPath, patch, 'utf8');
        const summary = summarizePatch(patch);
        const count = summary.files.length;
        this.post('patch', `Saved patch with ${count} file(s), +${summary.additions}/-${summary.deletions}.`);
        this.post('patchState', `Patch available (${count} file${count === 1 ? '' : 's'}, +${summary.additions}/-${summary.deletions})`);
        this.post('patchDetails', summary);
      }
      this.post('status', 'Ready');
    });
  }

  async copyLastPatch() {
    if (!fs.existsSync(this.lastPatchPath)) return vscode.window.showWarningMessage('No Hermes patch has been saved yet.');
    await vscode.env.clipboard.writeText(await fsp.readFile(this.lastPatchPath, 'utf8'));
    vscode.window.showInformationMessage('Copied last Hermes patch to clipboard.');
  }

  async previewLastPatch() {
    const workspace = getWorkspaceFolder();
    if (!workspace) return vscode.window.showWarningMessage('Open a workspace before previewing a patch.');
    if (!fs.existsSync(this.lastPatchPath)) return vscode.window.showWarningMessage('No Hermes patch has been saved yet.');
    const patchText = await fsp.readFile(this.lastPatchPath, 'utf8');
    const parsed = parsePatch(patchText);
    if (!parsed.length) {
      const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(this.lastPatchPath));
      await vscode.window.showTextDocument(doc);
      return;
    }
    const previewRoot = path.join(this.context.globalStorageUri.fsPath, 'preview', String(Date.now()));
    await fsp.mkdir(previewRoot, { recursive: true });
    let opened = 0;
    for (const filePatch of parsed) {
      const rel = filePatch.newPath;
      const originalPath = safeJoin(workspace.uri.fsPath, filePatch.oldPath === '/dev/null' ? rel : (filePatch.oldPath || rel));
      if (!fs.existsSync(originalPath)) continue;
      const original = await fsp.readFile(originalPath, 'utf8');
      const proposed = applyPatchToContent(original, filePatch);
      const proposedPath = safeJoin(previewRoot, rel);
      await fsp.mkdir(path.dirname(proposedPath), { recursive: true });
      await fsp.writeFile(proposedPath, proposed, 'utf8');
      await vscode.commands.executeCommand('vscode.diff', vscode.Uri.file(originalPath), vscode.Uri.file(proposedPath), `Hermes Preview: ${rel}`);
      opened += 1;
    }
    if (!opened) {
      const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(this.lastPatchPath));
      await vscode.window.showTextDocument(doc);
    }
  }

  async applyLastPatch() {
    const workspace = getWorkspaceFolder();
    if (!workspace) return vscode.window.showWarningMessage('Open a workspace before applying a patch.');
    if (!fs.existsSync(this.lastPatchPath)) return vscode.window.showWarningMessage('No Hermes patch has been saved yet.');
    const check = await runCommand('git', ['apply', '--check', this.lastPatchPath], workspace.uri.fsPath, 20000).catch((err) => ({ code: 1, stderr: String(err), stdout: '' }));
    if (check.code !== 0) {
      vscode.window.showErrorMessage(`Patch failed git apply --check: ${check.stderr || check.stdout}`);
      return;
    }
    const choice = await vscode.window.showWarningMessage('Apply the last Hermes patch to this workspace?', { modal: true }, 'Apply Patch');
    if (choice !== 'Apply Patch') return;
    const apply = await runCommand('git', ['apply', '--whitespace=nowarn', this.lastPatchPath], workspace.uri.fsPath, 20000).catch((err) => ({ code: 1, stderr: String(err), stdout: '' }));
    if (apply.code !== 0) vscode.window.showErrorMessage(`Patch apply failed: ${apply.stderr || apply.stdout}`);
    else {
      vscode.window.showInformationMessage('Hermes patch applied.');
      this.post('postApply', 'Patch applied. You can run tests or revert the patch from the review panel.');
    }
  }

  async discardLastPatch() {
    if (!fs.existsSync(this.lastPatchPath)) return vscode.window.showWarningMessage('No Hermes patch has been saved yet.');
    await fsp.rm(this.lastPatchPath, { force: true });
    this.post('patchState', 'No patch yet');
    this.post('patchDetails', null);
    this.post('patchCleared', 'Discarded saved patch.');
    vscode.window.showInformationMessage('Discarded last Hermes patch.');
  }

  async revertLastPatch() {
    const workspace = getWorkspaceFolder();
    if (!workspace) return vscode.window.showWarningMessage('Open a workspace before reverting a patch.');
    if (!fs.existsSync(this.lastPatchPath)) return vscode.window.showWarningMessage('No Hermes patch has been saved yet.');
    const check = await runCommand('git', ['apply', '-R', '--check', this.lastPatchPath], workspace.uri.fsPath, 20000).catch((err) => ({ code: 1, stderr: String(err), stdout: '' }));
    if (check.code !== 0) return vscode.window.showErrorMessage(`Patch cannot be reverted cleanly: ${check.stderr || check.stdout}`);
    const choice = await vscode.window.showWarningMessage('Revert the last Hermes patch from this workspace?', { modal: true }, 'Revert Patch');
    if (choice !== 'Revert Patch') return;
    const revert = await runCommand('git', ['apply', '-R', '--whitespace=nowarn', this.lastPatchPath], workspace.uri.fsPath, 20000).catch((err) => ({ code: 1, stderr: String(err), stdout: '' }));
    if (revert.code !== 0) vscode.window.showErrorMessage(`Patch revert failed: ${revert.stderr || revert.stdout}`);
    else vscode.window.showInformationMessage('Hermes patch reverted.');
  }

  async runDetectedTests() {
    const workspace = getWorkspaceFolder();
    if (!workspace) return vscode.window.showWarningMessage('Open a workspace before running tests.');
    const command = await detectTestCommand(workspace.uri.fsPath);
    const terminal = vscode.window.createTerminal({ name: 'Hermes Tests', cwd: workspace.uri.fsPath });
    terminal.show();
    terminal.sendText(`# Hermes Code detected test command`);
    terminal.sendText(command);
    this.post('tool', `Running tests: ${command}`);
  }

  async runTestsAndAnalyze() {
    const workspace = getWorkspaceFolder();
    if (!workspace) return vscode.window.showWarningMessage('Open a workspace before running tests.');
    const cwd = workspace.uri.fsPath;
    const command = await detectTestCommand(cwd);
    this.post('status', `Running tests: ${command}`);
    this.post('tool', `Running captured test command: ${command}`);
    this.output.appendLine(`\n[${new Date().toISOString()}] Hermes captured tests: ${command}`);
    const result = await runShellCommand(command, cwd, getMaxTestOutputChars()).catch((error) => ({ code: 1, stdout: '', stderr: error.message || String(error) }));
    const combined = [result.stdout, result.stderr].filter(Boolean).join('\n');
    this.output.appendLine(combined || '(test command produced no output)');
    const status = result.code === 0 ? 'passed' : `failed with exit ${result.code}`;
    this.post('tool', `Tests ${status}: ${command}`);
    const prompt = `The detected test command \`${command}\` ${status}. Analyze this output, identify likely root cause, and recommend the smallest next fix. If a code change is needed, return a unified diff.\n\n\`\`\`text\n${truncate(combined || '(no output)', getMaxTestOutputChars())}\n\`\`\``;
    await this.ask(prompt, { includeFile: true, includeSelection: true, includeDiagnostics: true, includeGitDiff: true, includeInstructions: true, wantsPatch: result.code !== 0, mode: result.code === 0 ? 'test' : 'debug' });
  }

  html() {
    const nonce = String(Date.now());
    return `<!DOCTYPE html><html><head><meta charset="UTF-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';"><style>
      :root { color-scheme: dark light; --hermes-accent: var(--vscode-button-background); --hermes-accent-fg: var(--vscode-button-foreground); --hermes-card: color-mix(in srgb, var(--vscode-editor-background) 86%, transparent); --hermes-soft-border: color-mix(in srgb, var(--vscode-sideBar-border) 68%, transparent); --hermes-muted: color-mix(in srgb, var(--vscode-foreground) 68%, transparent); }
      * { box-sizing: border-box; }
      body { margin: 0; min-height: 100vh; font-family: var(--vscode-font-family); color: var(--vscode-foreground); background: radial-gradient(circle at 20% -10%, color-mix(in srgb, var(--hermes-accent) 18%, transparent), transparent 34%), linear-gradient(180deg, color-mix(in srgb, var(--vscode-sideBar-background) 92%, var(--hermes-accent) 8%), var(--vscode-sideBar-background)); }
      header { padding: 14px 14px 12px; border-bottom: 1px solid var(--hermes-soft-border); display:flex; gap:10px; align-items:center; justify-content:space-between; backdrop-filter: blur(18px); }
      .brand { min-width:0; display:flex; gap:10px; align-items:center; }
      .orb { width:34px; height:34px; border-radius:12px; display:grid; place-items:center; color:var(--hermes-accent-fg); background: linear-gradient(135deg, var(--hermes-accent), color-mix(in srgb, var(--hermes-accent) 58%, #8b5cf6)); box-shadow: 0 10px 28px color-mix(in srgb, var(--hermes-accent) 26%, transparent); }
      .title { font-size: 13px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
      .subtitle { margin-top:2px; font-size: 11px; color: var(--hermes-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .status { max-width:42%; color: var(--hermes-muted); background: color-mix(in srgb, var(--vscode-input-background) 78%, transparent); border:1px solid var(--hermes-soft-border); border-radius:999px; padding:5px 8px; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      main { padding: 14px 12px 260px; display: flex; flex-direction: column; gap: 12px; height: 100vh; overflow-y: auto; scroll-behavior: smooth; }
      .msg { position:relative; border: 1px solid var(--hermes-soft-border); border-radius: 16px; padding: 12px 13px; white-space: pre-wrap; line-height: 1.48; word-break: break-word; box-shadow: 0 6px 24px rgba(0,0,0,.10); }
      .msg::before { content: attr(data-role); display:block; margin-bottom:7px; font-size:10px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--hermes-muted); }
      .user { margin-left:18px; background: linear-gradient(135deg, color-mix(in srgb, var(--hermes-accent) 26%, var(--vscode-input-background)), var(--vscode-input-background)); border-color: color-mix(in srgb, var(--hermes-accent) 38%, var(--hermes-soft-border)); }
      .assistant { margin-right:10px; background: var(--hermes-card); }
      .error { border-color: var(--vscode-errorForeground); color: var(--vscode-errorForeground); background: color-mix(in srgb, var(--vscode-errorForeground) 10%, var(--vscode-editor-background)); }
      .meta { color: var(--hermes-muted); font-size: 12px; background: color-mix(in srgb, var(--vscode-input-background) 60%, transparent); border-style:dashed; }
      footer { position: fixed; bottom: 0; left: 0; right: 0; padding: 10px 10px 12px; background: color-mix(in srgb, var(--vscode-sideBar-background) 88%, transparent); border-top: 1px solid var(--hermes-soft-border); backdrop-filter: blur(18px); box-shadow: 0 -12px 34px rgba(0,0,0,.18); }
      textarea { width: 100%; min-height: 74px; resize: vertical; box-sizing: border-box; color: var(--vscode-input-foreground); background: color-mix(in srgb, var(--vscode-input-background) 94%, var(--hermes-accent) 6%); border: 1px solid var(--hermes-soft-border); border-radius: 14px; padding: 11px 12px; outline: none; line-height:1.35; box-shadow: inset 0 1px 0 rgba(255,255,255,.03); }
      textarea:focus { border-color: color-mix(in srgb, var(--hermes-accent) 65%, var(--hermes-soft-border)); box-shadow: 0 0 0 2px color-mix(in srgb, var(--hermes-accent) 18%, transparent); }
      button { color: var(--vscode-button-foreground); background: var(--vscode-button-background); border: 0; border-radius: 999px; padding: 7px 11px; margin-top: 0; cursor:pointer; font-weight:600; }
      button:hover { filter: brightness(1.08); }
      button.secondary { background: color-mix(in srgb, var(--vscode-button-secondaryBackground) 82%, transparent); color: var(--vscode-button-secondaryForeground); border: 1px solid var(--hermes-soft-border); }
      label { font-size: 12px; color: var(--hermes-muted); margin-right: 0; white-space: nowrap; display:inline-flex; gap:4px; align-items:center; padding:4px 7px; border:1px solid var(--hermes-soft-border); border-radius:999px; background: color-mix(in srgb, var(--vscode-input-background) 66%, transparent); }
      .actions, .toggles { display:flex; gap:7px; flex-wrap:wrap; align-items:center; }
      .toggles { margin-top: 8px; justify-content: space-between; }
      .toggle-group { display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
      .actions { margin-top:9px; }
      .patch-pill { margin-left:auto; color: var(--hermes-muted); font-size: 11px; border:1px solid var(--hermes-soft-border); border-radius:999px; padding:4px 8px; background: color-mix(in srgb, var(--vscode-editor-background) 72%, transparent); }
      .composer-top { display:flex; gap:8px; align-items:center; margin-bottom:8px; }
      select { min-width: 102px; color: var(--vscode-dropdown-foreground); background: var(--vscode-dropdown-background); border:1px solid var(--vscode-dropdown-border); border-radius:999px; padding:6px 8px; }
      .prompt-buttons { display:flex; gap:6px; flex-wrap:wrap; margin: 10px 0 2px; }
      .prompt-buttons button { font-size: 11px; padding: 6px 8px; }
      pre { position: relative; margin: 10px 0; padding: 12px; border-radius: 12px; overflow:auto; background: var(--vscode-textCodeBlock-background); border:1px solid var(--hermes-soft-border); }
      code { font-family: var(--vscode-editor-font-family); font-size: var(--vscode-editor-font-size); }
      .copy { position:absolute; top:8px; right:8px; font-size:10px; padding:4px 7px; opacity:.84; }
      .msg-actions { display:flex; justify-content:flex-end; margin-top:8px; gap:6px; }
      .msg-actions button { font-size:10px; padding:4px 7px; }
      details.tool-event { border:1px dashed var(--hermes-soft-border); border-radius: 12px; padding: 8px 10px; background: color-mix(in srgb, var(--vscode-input-background) 58%, transparent); color: var(--hermes-muted); }
      details.tool-event summary { cursor:pointer; font-weight:700; color: var(--vscode-foreground); }
      .patch-panel, .context-panel { display:none; border:1px solid color-mix(in srgb, var(--hermes-accent) 34%, var(--hermes-soft-border)); border-radius:16px; padding:10px; margin-bottom:9px; background: color-mix(in srgb, var(--hermes-accent) 10%, var(--vscode-editor-background)); }
      .patch-panel.visible, .context-panel.visible { display:block; }
      .patch-panel strong, .context-panel strong { display:block; margin-bottom:7px; }
      .patch-panel .actions { margin-top:0; }
      .panel-summary { color: var(--hermes-muted); font-size: 12px; line-height:1.45; margin-bottom:8px; }
      .file-list { display:flex; flex-direction:column; gap:5px; max-height:96px; overflow:auto; margin-bottom:8px; }
      .file-row { display:flex; justify-content:space-between; gap:8px; border:1px solid var(--hermes-soft-border); border-radius:10px; padding:5px 7px; background: color-mix(in srgb, var(--vscode-editor-background) 70%, transparent); font-size:11px; }
      .file-row span:first-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      .delta-plus { color: var(--vscode-gitDecoration-addedResourceForeground, #3fb950); }
      .delta-minus { color: var(--vscode-gitDecoration-deletedResourceForeground, #f85149); }
    </style></head><body><header><div class="brand"><div class="orb">✦</div><div><div class="title">Hermes Code</div><div class="subtitle">Workspace-aware coding copilot</div></div></div><span class="status" id="status">Ready</span></header><main id="messages"><div class="msg assistant" data-role="Hermes"><p>Ask Hermes to inspect, explain, edit, or review this workspace. Use the context menu for selection-aware commands.</p><div class="prompt-buttons"><button data-prompt="Review the current diff for risk, likely breakage, missing tests, and concrete fixes." class="secondary">Review diff</button><button data-prompt="Explain the active file and call out important risks or improvements." class="secondary">Explain file</button><button data-prompt="Find missing tests or weak coverage for the selected code or active file." class="secondary">Find tests</button></div></div></main><footer><div id="contextPanel" class="context-panel"><strong>Context inspector</strong><div id="contextSummary" class="panel-summary">Click Context to inspect what Hermes will see.</div></div><div id="patchPanel" class="patch-panel"><strong>Patch review</strong><div id="patchSummary" class="panel-summary">Patch summary unavailable.</div><div id="patchFiles" class="file-list"></div><div class="actions"><button class="secondary" id="preview">Preview</button><button id="apply">Apply</button><button class="secondary" id="copy">Copy</button><button class="secondary" id="discard">Discard</button><button class="secondary" id="runTests">Run tests</button><button class="secondary" id="runAnalyze">Run + analyze</button><button class="secondary" id="revert">Revert</button></div></div><div class="composer-top"><select id="mode"><option value="ask">Ask</option><option value="explain">Explain</option><option value="edit">Edit</option><option value="review">Review</option><option value="test">Test</option><option value="debug">Debug</option><option value="refactor">Refactor</option><option value="security">Security</option><option value="commit">Commit</option></select><textarea id="input" placeholder="Ask Hermes to edit, explain, test, or review…  ⌘/Ctrl+Enter"></textarea></div><div class="toggles"><div class="toggle-group"><label><input id="ctxFile" type="checkbox" checked> file</label><label><input id="ctxSel" type="checkbox" checked> selection</label><label><input id="ctxDiag" type="checkbox" checked> diagnostics</label><label><input id="ctxGit" type="checkbox"> git diff</label><label><input id="ctxInst" type="checkbox" checked> instructions</label></div><span class="patch-pill" id="patchState">No patch yet</span></div><div class="actions"><button id="send">Send</button><button class="secondary" id="review">Review diff</button><button class="secondary" id="diag">Diagnostics</button><button class="secondary" id="term">Terminal</button><button class="secondary" id="context">Context</button><button class="secondary" id="stop">Stop</button><button class="secondary" id="clear">Clear</button><button class="secondary" id="config">Config</button></div></footer><script nonce="${nonce}">
      const vscode = acquireVsCodeApi(); const messages = document.getElementById('messages'); const input = document.getElementById('input'); const status = document.getElementById('status'); const patchState = document.getElementById('patchState'); const patchPanel = document.getElementById('patchPanel'); const patchSummary = document.getElementById('patchSummary'); const patchFiles = document.getElementById('patchFiles'); const contextPanel = document.getElementById('contextPanel'); const contextSummary = document.getElementById('contextSummary'); const modeSelect = document.getElementById('mode'); let partial;
      function escapeHtml(text){ return (text||'').replace(/[&<>]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[ch])); }
      function renderMarkdown(text){ const tick=String.fromCharCode(96); const fence=tick+tick+tick; const parts = String(text||'').split(new RegExp(fence+'([\\w.+-]*)\\n([\\s\\S]*?)'+fence,'g')); let html=''; for(let i=0;i<parts.length;i++){ if(i%3===0){ html += escapeHtml(parts[i]).replace(/^### (.*)$/gm,'<h3>$1</h3>').replace(/^## (.*)$/gm,'<h2>$1</h2>').replace(/^# (.*)$/gm,'<h1>$1</h1>').replace(new RegExp(tick+'([^'+tick+']+)'+tick,'g'),'<code>$1</code>').replace(/\n/g,'<br>'); } else if(i%3===1){ const lang=escapeHtml(parts[i]||'text'); const code=escapeHtml(parts[i+1]||''); html += '<pre><button class="copy" data-copy="code">Copy</button><code data-lang="'+lang+'">'+code+'</code></pre>'; i++; } } return html; }
      function roleLabel(cls){ if(cls==='user') return 'You'; if(cls==='assistant') return 'Hermes'; if(cls==='error') return 'Error'; return 'Event'; }
      function addAssistantActions(div){ const actions=document.createElement('div'); actions.className='msg-actions'; actions.innerHTML='<button class="secondary" data-copy="message">Copy</button><button class="secondary" data-retry="1">Retry</button>'; div.appendChild(actions); }
      function add(cls, text){ const div=document.createElement('div'); div.className='msg '+cls; div.dataset.role=roleLabel(cls); div.innerHTML=renderMarkdown(text); if(cls==='assistant') addAssistantActions(div); messages.appendChild(div); messages.scrollTop=messages.scrollHeight; return div; }
      function addTool(text){ const details=document.createElement('details'); details.className='tool-event'; details.innerHTML='<summary>🔧 '+escapeHtml(String(text).split('\n')[0]||'Hermes tool')+'</summary><div>'+renderMarkdown(text)+'</div>'; messages.appendChild(details); messages.scrollTop=messages.scrollHeight; }
      function options(){ return {mode: modeSelect.value, includeFile:ctxFile.checked, includeSelection:ctxSel.checked, includeDiagnostics:ctxDiag.checked, includeGitDiff:ctxGit.checked, includeInstructions:ctxInst.checked}; }
      function send(textOverride){ const text=textOverride || input.value; if(!text.trim()) return; input.value=''; vscode.postMessage({type:'ask', text, options: options()}); }
      function setPatchState(value){ patchState.textContent=value; patchPanel.classList.toggle('visible', value && !/^No patch/.test(value)); }
      function renderPatchDetails(value){ if(!value || !value.files || !value.files.length){ patchSummary.textContent='No patch metadata available.'; patchFiles.innerHTML=''; return; } patchSummary.textContent=value.files.length+' file(s), '+value.hunks+' hunk(s), +'+value.additions+'/-'+value.deletions; patchFiles.innerHTML=value.files.map(f=>'<div class="file-row"><span>'+escapeHtml(f.path||'unknown')+'</span><span><span class="delta-plus">+'+f.additions+'</span> <span class="delta-minus">-'+f.deletions+'</span> · '+f.hunks+' h</span></div>').join(''); }
      function renderContextSummary(value){ if(!value){ contextSummary.textContent='No context summary available.'; return; } const included=Object.entries(value.included||{}).filter(([,on])=>on).map(([name])=>name).join(', ') || 'none'; contextSummary.innerHTML='<div><b>Workspace</b>: '+escapeHtml(value.workspace||'none')+'</div><div><b>Active file</b>: '+escapeHtml(value.activeFile||'none')+(value.dirty?' <em>(unsaved)</em>':'')+'</div><div><b>Language</b>: '+escapeHtml(value.language||'n/a')+'</div><div><b>Selection</b>: '+value.selectionLines+' line(s), '+value.selectionChars+' chars</div><div><b>Diagnostics</b>: '+value.diagnostics+'</div><div><b>Git status files</b>: '+value.gitFiles+'</div><div><b>Instructions</b>: '+escapeHtml((value.instructionFiles||[]).join(', ')||'none')+'</div><div><b>Included</b>: '+escapeHtml(included)+'</div><div><b>Budget</b>: '+value.maxContextChars+' chars</div>'; contextPanel.classList.add('visible'); }
      document.getElementById('send').onclick=()=>send();
      document.getElementById('review').onclick=()=>vscode.postMessage({type:'reviewDiff'});
      document.getElementById('diag').onclick=()=>vscode.postMessage({type:'diagnostics'});
      document.getElementById('term').onclick=()=>vscode.postMessage({type:'terminal'});
      document.getElementById('context').onclick=()=>vscode.postMessage({type:'inspectContext', options: options()});
      document.getElementById('preview').onclick=()=>vscode.postMessage({type:'previewPatch'});
      document.getElementById('apply').onclick=()=>vscode.postMessage({type:'applyPatch'});
      document.getElementById('copy').onclick=()=>vscode.postMessage({type:'copyPatch'});
      document.getElementById('discard').onclick=()=>vscode.postMessage({type:'discardPatch'});
      document.getElementById('runTests').onclick=()=>vscode.postMessage({type:'runTests'});
      document.getElementById('runAnalyze').onclick=()=>vscode.postMessage({type:'runTestsAndAnalyze'});
      document.getElementById('revert').onclick=()=>vscode.postMessage({type:'revertPatch'});
      document.getElementById('stop').onclick=()=>vscode.postMessage({type:'stop'});
      document.getElementById('clear').onclick=()=>vscode.postMessage({type:'clear'});
      document.getElementById('config').onclick=()=>vscode.postMessage({type:'configure'});
      input.addEventListener('keydown', e=>{ if ((e.metaKey||e.ctrlKey) && e.key === 'Enter') send(); });
      document.addEventListener('click', e=>{ const prompt=e.target.dataset.prompt; if(prompt) send(prompt); if(e.target.dataset.copy==='message'){ navigator.clipboard.writeText(e.target.closest('.msg').innerText.replace(/^Hermes\n/,'').replace(/Copy\s*Retry$/,'')); } if(e.target.dataset.copy==='code'){ navigator.clipboard.writeText(e.target.parentElement.querySelector('code').innerText); } if(e.target.dataset.retry){ const last=[...messages.querySelectorAll('.user')].pop(); if(last) send(last.innerText.replace(/^You\n/,'')); } });
      window.addEventListener('message', event=>{ const {type,value}=event.data; if(type==='restore'){ messages.innerHTML=''; if(!value.length) add('assistant','Ask Hermes to inspect, explain, edit, or review this workspace.'); value.forEach(m=>add(m.role === 'error' ? 'error' : m.role, m.text)); } if(type==='patchState') setPatchState(value); if(type==='patchDetails') renderPatchDetails(value); if(type==='contextSummary') renderContextSummary(value); if(type==='status') status.textContent=value; if(type==='user') { partial=undefined; add('user', value); } if(type==='partial') { if(!partial) partial=add('assistant', value); else partial.innerHTML=renderMarkdown(value); } if(type==='assistant') { if(partial) { partial.innerHTML=renderMarkdown(value); addAssistantActions(partial); } else add('assistant', value); partial=undefined; } if(type==='error') add('error', value); if(type==='tool') addTool(value); if(type==='patch') { addTool(value+' Use the Patch review panel to preview/apply/copy/discard.'); setPatchState(value); } if(type==='postApply') addTool(value); if(type==='patchCleared') addTool(value); });
      vscode.postMessage({type:'ready'});
    </script></body></html>`;
  }
}

function getWorkspaceFolder() {
  const editor = vscode.window.activeTextEditor;
  if (editor) return vscode.workspace.getWorkspaceFolder(editor.document.uri);
  return vscode.workspace.workspaceFolders?.[0];
}

function getMaxContextChars() {
  return vscode.workspace.getConfiguration('hermesCode').get('maxContextChars', 32000);
}

function getMaxTestOutputChars() {
  return vscode.workspace.getConfiguration('hermesCode').get('maxTestOutputChars', 30000);
}

async function collectEditorContext(options = {}, output) {
  const max = getMaxContextChars();
  const editor = vscode.window.activeTextEditor;
  const parts = [];
  const workspace = getWorkspaceFolder();
  if (workspace) parts.push(`Workspace: ${workspace.uri.fsPath}`);
  if (workspace && options.includeInstructions !== false) {
    const instructions = await readInstructionFiles(workspace.uri.fsPath, output);
    if (instructions) parts.push(instructions);
  }
  if (editor) {
    const doc = editor.document;
    parts.push(`Active file: ${doc.uri.fsPath}`);
    parts.push(`Language: ${doc.languageId}`);
    const selected = getSelectionText(editor);
    if ((options.forceSelection || options.includeSelection) && selected) {
      parts.push(`## Selected text\n\n\`\`\`${doc.languageId}\n${truncate(selected, Math.floor(max / 3))}\n\`\`\``);
    }
    if (options.includeFile !== false && !(options.forceSelection && selected)) {
      parts.push(`## Active file content${doc.isDirty ? ' (unsaved buffer)' : ''}\n\n\`\`\`${doc.languageId}\n${truncate(doc.getText(), Math.floor(max / 2))}\n\`\`\``);
    }
  }
  if (options.includeDiagnostics !== false && vscode.workspace.getConfiguration('hermesCode').get('includeDiagnosticsByDefault', true)) {
    const diagnostics = diagnosticsForActiveEditor();
    if (diagnostics) parts.push(diagnostics);
  }
  if (workspace && options.includeGitDiff) {
    const diff = await getGitDiff(workspace.uri.fsPath).catch(() => '');
    if (diff.trim()) parts.push(`## Current Git diff\n\n\`\`\`diff\n${truncate(diff, Math.floor(max / 2))}\n\`\`\``);
  }
  if (options.extraContext) parts.push(options.extraContext);
  return truncate(parts.join('\n\n'), max);
}

const INSTRUCTION_FILE_NAMES = ['AGENTS.md', 'CLAUDE.md', 'GEMINI.md', '.cursorrules', path.join('openspec', 'AGENTS.md')];

async function collectContextSummary(options = {}) {
  const workspace = getWorkspaceFolder();
  const editor = vscode.window.activeTextEditor;
  const summary = {
    workspace: workspace?.uri.fsPath || null,
    activeFile: editor?.document.uri.fsPath || null,
    language: editor?.document.languageId || null,
    dirty: Boolean(editor?.document.isDirty),
    selectionChars: 0,
    selectionLines: 0,
    diagnostics: 0,
    gitFiles: 0,
    instructionFiles: [],
    included: {
      file: options.includeFile !== false,
      selection: options.includeSelection !== false,
      diagnostics: options.includeDiagnostics !== false && vscode.workspace.getConfiguration('hermesCode').get('includeDiagnosticsByDefault', true),
      gitDiff: Boolean(options.includeGitDiff),
      instructions: options.includeInstructions !== false
    },
    maxContextChars: getMaxContextChars()
  };
  if (editor && !editor.selection.isEmpty) {
    const selected = getSelectionText(editor);
    summary.selectionChars = selected.length;
    summary.selectionLines = selected.split(/\r?\n/).length;
  }
  if (editor) summary.diagnostics = vscode.languages.getDiagnostics(editor.document.uri).length;
  if (workspace) {
    summary.instructionFiles = await listInstructionFiles(workspace.uri.fsPath);
    const status = await runCommand('git', ['status', '--short'], workspace.uri.fsPath, 20000).catch(() => ({ stdout: '' }));
    summary.gitFiles = status.stdout.split('\n').filter((line) => line.trim()).length;
  }
  return summary;
}

async function listInstructionFiles(root) {
  const found = [];
  for (const name of INSTRUCTION_FILE_NAMES) {
    try {
      const stat = await fsp.stat(path.join(root, name));
      if (stat.isFile() && stat.size <= 80_000) found.push(name);
    } catch {}
  }
  return found;
}

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

async function readInstructionFiles(root, output) {
  const names = INSTRUCTION_FILE_NAMES;
  const chunks = [];
  for (const name of names) {
    const file = path.join(root, name);
    try {
      const stat = await fsp.stat(file);
      if (!stat.isFile() || stat.size > 80_000) continue;
      chunks.push(`### ${name}\n\n${await fsp.readFile(file, 'utf8')}`);
    } catch (err) {
      if (err && err.code !== 'ENOENT') output?.appendLine(`Instruction read skipped for ${name}: ${err.message}`);
    }
  }
  return chunks.length ? `## Workspace instruction files\n\n${chunks.join('\n\n')}` : '';
}

async function getGitDiff(cwd) {
  const status = await runCommand('git', ['status', '--short'], cwd, 20000).catch(() => ({ stdout: '' }));
  const unstaged = await runCommand('git', ['diff', '--no-ext-diff', '--', '.'], cwd, 200000).catch(() => ({ stdout: '' }));
  const staged = await runCommand('git', ['diff', '--cached', '--no-ext-diff', '--', '.'], cwd, 200000).catch(() => ({ stdout: '' }));
  return [`# git status --short\n${status.stdout}`, staged.stdout ? `# staged diff\n${staged.stdout}` : '', unstaged.stdout ? `# unstaged diff\n${unstaged.stdout}` : ''].filter(Boolean).join('\n\n');
}

function buildPrompt(userText, contextBlock, options = {}) {
  const mode = options.mode || 'ask';
  const modeInstruction = modeInstructions(mode);
  const patchInstruction = options.wantsPatch || ['edit', 'debug', 'refactor', 'test', 'security'].includes(mode)
    ? '\nIf code edits are useful, return them as a unified diff in a ```diff fenced block. Keep explanations short and include verification commands.'
    : '\nWhen code edits are useful, prefer a short explanation plus a unified diff in a ```diff fenced block.';
  return `You are Hermes running inside a VS Code/Cursor extension. Use the supplied editor context. Be concise, operational, and careful with unsaved buffers. Do not expose secrets.\n${modeInstruction}${patchInstruction}\n\n${contextBlock}\n\n## User request\n\n${userText}`;
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

function workspaceSessionName(cwd) {
  const hash = crypto.createHash('sha1').update(cwd || 'workspace').digest('hex').slice(0, 10);
  return `vscode-${path.basename(cwd || 'workspace')}-${hash}`;
}

function promptSummary(prompt) {
  const marker = '## User request';
  const idx = prompt.lastIndexOf(marker);
  return idx >= 0 ? prompt.slice(idx + marker.length).trim() : truncate(prompt, 800);
}

function getSelectionText(editor) {
  if (!editor || editor.selection.isEmpty) return '';
  return editor.document.getText(editor.selection);
}

function diagnosticsForActiveEditor() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return '';
  const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
  if (!diagnostics.length) return '';
  const body = diagnostics.map((d) => {
    const start = `${d.range.start.line + 1}:${d.range.start.character + 1}`;
    const severity = ['Error', 'Warning', 'Info', 'Hint'][d.severity] || 'Diagnostic';
    return `- ${severity} ${start}: ${d.message}`;
  }).join('\n');
  return `## Active file diagnostics\n\n${body}`;
}

function truncate(text, max) {
  if (!text || text.length <= max) return text || '';
  return text.slice(0, max) + `\n\n...[truncated ${text.length - max} chars]`;
}

async function detectTestCommand(cwd) {
  const exists = async (name) => {
    try { await fsp.stat(path.join(cwd, name)); return true; } catch { return false; }
  };
  if (await exists('package.json')) return 'npm test';
  if (await exists('pyproject.toml') || await exists('pytest.ini')) return 'python -m pytest';
  if (await exists('Package.swift')) return 'swift test';
  if (await exists('go.mod')) return 'go test ./...';
  if (await exists('Cargo.toml')) return 'cargo test';
  if (await exists('Makefile')) return 'make test';
  return 'echo "No standard test command detected. Edit this command and press Enter."';
}

function runCommand(command, args, cwd, maxBuffer = 100000) {
  return new Promise((resolve, reject) => {
    cp.execFile(command, args, { cwd, maxBuffer }, (error, stdout, stderr) => {
      const result = { code: error?.code || 0, stdout, stderr };
      if (error && error.code === undefined) reject(error);
      else resolve(result);
    });
  });
}

function runShellCommand(command, cwd, maxBuffer = 100000) {
  return new Promise((resolve, reject) => {
    cp.exec(command, { cwd, maxBuffer }, (error, stdout, stderr) => {
      const result = { code: error?.code || 0, stdout, stderr };
      if (error && error.code === undefined) reject(error);
      else resolve(result);
    });
  });
}

module.exports = { activate, deactivate };
