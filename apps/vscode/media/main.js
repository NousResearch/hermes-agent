const vscode = acquireVsCodeApi();
const messages = document.getElementById('messages');
const input = document.getElementById('input');
const status = document.getElementById('status');
const patchState = document.getElementById('patchState');
const patchPanel = document.getElementById('patchPanel');
const patchSummary = document.getElementById('patchSummary');
const patchFiles = document.getElementById('patchFiles');
const contextPanel = document.getElementById('contextPanel');
const contextSummary = document.getElementById('contextSummary');
const modeSelect = document.getElementById('mode');
const footer = document.querySelector('footer');
const codeBlocks = new Map();
let partial;
let codeBlockCounter = 0;

function escapeHtml(text) {
  return (text || '').replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]));
}

function isDiffBlock(lang, code) {
  const value = String(code || '');
  return /diff|patch/i.test(lang || '') || /^diff --git /m.test(value) || /^@@\s/m.test(value);
}

function isRunnableBlock(lang, code) {
  if (isDiffBlock(lang, code)) return false;
  const normalizedLang = String(lang || '').toLowerCase();
  if (/^(sh|shell|bash|zsh|fish|powershell|pwsh|ps1|terminal|console|cmd|bat|batch)$/.test(normalizedLang)) return true;
  if (normalizedLang && !/^(text|txt|plain|plaintext)$/.test(normalizedLang)) return false;
  const lines = String(code || '').split('\n').map((line) => line.trim()).filter(Boolean);
  if (!lines.length || lines.length > 20) return false;
  return lines.every((line) =>
    !line.startsWith('#') &&
    !/[{};]/.test(line) &&
    /^(cd\s|npm\s|pnpm\s|yarn\s|bun\s|node\s|python\b|python3\b|pytest\b|poetry\s|uv\s|pip\s|make\b|go\s|cargo\s|swift\s|docker\s|kubectl\s|terraform\s|ansible\s|git\s|gh\s|hermes\s|npx\s|\.\/|[A-Za-z0-9_./~-]+\s+[-\w./~])/.test(line)
  );
}

function renderDiff(code) {
  return String(code || '').split('\n').map((line) => {
    let cls = 'diff-context';
    if (line.startsWith('+') && !line.startsWith('+++')) cls = 'diff-add';
    else if (line.startsWith('-') && !line.startsWith('---')) cls = 'diff-del';
    else if (line.startsWith('@@')) cls = 'diff-hunk';
    else if (/^(diff --git|index |--- |\+\+\+ )/.test(line)) cls = 'diff-meta';
    return '<span class="' + cls + '">' + escapeHtml(line || ' ') + '</span>';
  }).join('\n');
}

function renderMarkdown(text) {
  const tick = String.fromCharCode(96);
  const fence = tick + tick + tick;
  const parts = String(text || '').split(new RegExp(fence + '([\\w.+-]*)\\n([\\s\\S]*?)' + fence, 'g'));
  let html = '';
  for (let i = 0; i < parts.length; i++) {
    if (i % 3 === 0) {
      html += escapeHtml(parts[i])
        .replace(/^### (.*)$/gm, '<h3>$1</h3>')
        .replace(/^## (.*)$/gm, '<h2>$1</h2>')
        .replace(/^# (.*)$/gm, '<h1>$1</h1>')
        .replace(new RegExp(tick + '([^' + tick + ']+)' + tick, 'g'), '<code>$1</code>')
        .replace(/\n/g, '<br>');
    } else if (i % 3 === 1) {
      const lang = parts[i] || 'text';
      const code = parts[i + 1] || '';
      const id = 'code-' + (++codeBlockCounter);
      const diff = isDiffBlock(lang, code);
      const runnable = isRunnableBlock(lang, code);
      codeBlocks.set(id, code);
      const rendered = diff ? renderDiff(code) : escapeHtml(code);
      const runButton = runnable ? '<button class="run-inline" data-run-code="' + id + '">Run</button>' : '';
      const applyButton = diff ? '<button class="apply-inline" data-apply-code="' + id + '">Apply</button>' : '';
      html += '<pre class="' + (diff ? 'diff-block' : runnable ? 'command-block' : '') + '"><div class="code-actions"><button class="copy" data-copy="code" data-code-id="' + id + '">Copy</button>' + runButton + applyButton + '</div><code data-lang="' + escapeHtml(lang) + '">' + rendered + '</code></pre>';
      i++;
    }
  }
  return html;
}

function roleLabel(cls) {
  if (cls === 'user') return 'You';
  if (cls === 'assistant') return 'Hermes';
  if (cls === 'error') return 'Error';
  return 'Event';
}

function scrollToBottom() {
  requestAnimationFrame(() => { messages.scrollTop = messages.scrollHeight; });
}

function syncFooterHeight() {
  const height = footer ? Math.ceil(footer.getBoundingClientRect().height) : 0;
  document.documentElement.style.setProperty('--footer-height', height + 'px');
}

function addAssistantActions(div) {
  const actions = document.createElement('div');
  actions.className = 'msg-actions';
  actions.innerHTML = '<button class="secondary" data-copy="message">Copy</button><button class="secondary" data-retry="1">Retry</button>';
  div.appendChild(actions);
}

function add(cls, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.dataset.role = roleLabel(cls);
  div.innerHTML = renderMarkdown(text);
  if (cls === 'assistant') addAssistantActions(div);
  messages.appendChild(div);
  scrollToBottom();
  return div;
}

function addTool(text) {
  const details = document.createElement('details');
  details.className = 'tool-event';
  details.innerHTML = '<summary>🔧 ' + escapeHtml(String(text).split('\n')[0] || 'Hermes tool') + '</summary><div>' + renderMarkdown(text) + '</div>';
  messages.appendChild(details);
  scrollToBottom();
}

function options() {
  return { mode: modeSelect.value, includeFile: ctxFile.checked, includeSelection: ctxSel.checked, includeDiagnostics: ctxDiag.checked, includeGitDiff: ctxGit.checked, includeInstructions: ctxInst.checked };
}

function send(textOverride) {
  const text = textOverride || input.value;
  if (!text.trim()) return;
  input.value = '';
  vscode.postMessage({ type: 'ask', text, options: options() });
}

function setPatchState(value) {
  patchState.textContent = value;
  patchPanel.classList.toggle('visible', value && !/^No patch/.test(value));
  syncFooterHeight();
  scrollToBottom();
}

function renderPatchDetails(value) {
  if (!value || !value.files || !value.files.length) {
    patchSummary.textContent = 'No patch metadata available.';
    patchFiles.innerHTML = '';
    syncFooterHeight();
    return;
  }
  patchSummary.textContent = value.files.length + ' file(s), ' + value.hunks + ' hunk(s), +' + value.additions + '/-' + value.deletions;
  patchFiles.innerHTML = value.files.map((f) => '<div class="file-row"><span>' + escapeHtml(f.path || 'unknown') + '</span><span><span class="delta-plus">+' + f.additions + '</span> <span class="delta-minus">-' + f.deletions + '</span> · ' + f.hunks + ' h</span></div>').join('');
  syncFooterHeight();
}

function renderContextSummary(value) {
  if (!value) {
    contextSummary.textContent = 'No context summary available.';
    syncFooterHeight();
    return;
  }
  const included = Object.entries(value.included || {}).filter(([, on]) => on).map(([name]) => name).join(', ') || 'none';
  contextSummary.innerHTML = '<div><b>Workspace</b>: ' + escapeHtml(value.workspace || 'none') + '</div><div><b>Active file</b>: ' + escapeHtml(value.activeFile || 'none') + (value.dirty ? ' <em>(unsaved)</em>' : '') + '</div><div><b>Language</b>: ' + escapeHtml(value.language || 'n/a') + '</div><div><b>Selection</b>: ' + value.selectionLines + ' line(s), ' + value.selectionChars + ' chars</div><div><b>Diagnostics</b>: ' + value.diagnostics + '</div><div><b>Git status files</b>: ' + value.gitFiles + '</div><div><b>Instructions</b>: ' + escapeHtml((value.instructionFiles || []).join(', ') || 'none') + '</div><div><b>Included</b>: ' + escapeHtml(included) + '</div><div><b>Budget</b>: ' + value.maxContextChars + ' chars</div>';
  contextPanel.classList.add('visible');
  syncFooterHeight();
  scrollToBottom();
}

document.getElementById('send').onclick = () => send();
document.getElementById('review').onclick = () => vscode.postMessage({ type: 'reviewDiff' });
document.getElementById('diag').onclick = () => vscode.postMessage({ type: 'diagnostics' });
document.getElementById('term').onclick = () => vscode.postMessage({ type: 'terminal' });
document.getElementById('context').onclick = () => vscode.postMessage({ type: 'inspectContext', options: options() });
document.getElementById('preview').onclick = () => vscode.postMessage({ type: 'previewPatch' });
document.getElementById('apply').onclick = () => vscode.postMessage({ type: 'applyPatch' });
document.getElementById('copy').onclick = () => vscode.postMessage({ type: 'copyPatch' });
document.getElementById('discard').onclick = () => vscode.postMessage({ type: 'discardPatch' });
document.getElementById('runTests').onclick = () => vscode.postMessage({ type: 'runTests' });
document.getElementById('runAnalyze').onclick = () => vscode.postMessage({ type: 'runTestsAndAnalyze' });
document.getElementById('revert').onclick = () => vscode.postMessage({ type: 'revertPatch' });
document.getElementById('stop').onclick = () => vscode.postMessage({ type: 'stop' });
document.getElementById('clear').onclick = () => vscode.postMessage({ type: 'clear' });
document.getElementById('config').onclick = () => vscode.postMessage({ type: 'configure' });
input.addEventListener('keydown', (e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') send(); });
document.addEventListener('click', (e) => {
  const prompt = e.target.dataset.prompt;
  if (prompt) send(prompt);
  if (e.target.dataset.copy === 'message') {
    navigator.clipboard.writeText(e.target.closest('.msg').innerText.replace(/^Hermes\n/, '').replace(/Copy\s*Retry$/, ''));
  }
  if (e.target.dataset.copy === 'code') {
    navigator.clipboard.writeText(codeBlocks.get(e.target.dataset.codeId) || e.target.closest('pre').querySelector('code').innerText);
  }
  if (e.target.dataset.applyCode) {
    vscode.postMessage({ type: 'applyInlinePatch', value: codeBlocks.get(e.target.dataset.applyCode) || '' });
  }
  if (e.target.dataset.runCode) {
    vscode.postMessage({ type: 'runInlineCommand', value: codeBlocks.get(e.target.dataset.runCode) || '' });
  }
  if (e.target.dataset.retry) {
    const last = [...messages.querySelectorAll('.user')].pop();
    if (last) send(last.innerText.replace(/^You\n/, ''));
  }
});
window.addEventListener('message', (event) => {
  const { type, value } = event.data;
  if (type === 'restore') {
    messages.innerHTML = '';
    codeBlocks.clear();
    if (!value.length) add('assistant', 'Ask Hermes to inspect, explain, edit, or review this workspace.');
    value.forEach((m) => add(m.role === 'error' ? 'error' : m.role, m.text));
  }
  if (type === 'patchState') setPatchState(value);
  if (type === 'patchDetails') renderPatchDetails(value);
  if (type === 'contextSummary') renderContextSummary(value);
  if (type === 'status') status.textContent = value;
  if (type === 'user') { partial = undefined; add('user', value); }
  if (type === 'partial') { if (!partial) partial = add('assistant', value); else partial.innerHTML = renderMarkdown(value); scrollToBottom(); }
  if (type === 'assistant') { if (partial) { partial.innerHTML = renderMarkdown(value); addAssistantActions(partial); } else add('assistant', value); partial = undefined; scrollToBottom(); }
  if (type === 'error') add('error', value);
  if (type === 'tool') addTool(value);
  if (type === 'patch') { addTool(value + ' Use the Patch review panel to preview/apply/copy/discard.'); setPatchState(value); }
  if (type === 'postApply') addTool(value);
  if (type === 'patchCleared') addTool(value);
});
if (footer && typeof ResizeObserver !== 'undefined') new ResizeObserver(syncFooterHeight).observe(footer);
window.addEventListener('resize', syncFooterHeight);
syncFooterHeight();
vscode.postMessage({ type: 'ready' });
