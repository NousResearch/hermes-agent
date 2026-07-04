/* hermes-monaco.js
 * Monaco editor surface for the Hermes AE Hub viewport.
 * Mounts into #monaco-host, loads from CDN, and bridges edits to vscode://.
 */
(function () {
  const HOST = document.getElementById('monaco-host');
  if (!HOST || HOST.dataset.ready === '1') return;
  HOST.dataset.ready = '1';
  const LABEL_ID = 'monaco-status-label';

  function setLabel(text) {
    let el = document.getElementById(LABEL_ID);
    if (!el) {
      el = document.createElement('div');
      el.id = LABEL_ID;
      el.className = 'panel-tag';
      el.style.cssText = 'position:absolute;right:10px;top:10px;color:var(--muted);font-size:10px;letter-spacing:0.12em;text-transform:uppercase;pointer-events:none;';
      HOST.parentElement.appendChild(el);
    }
    el.textContent = text || 'ready';
  }

  async function bootstrap() {
    if (!window.MonacoEnvironment) {
      window.MonacoEnvironment = {
        getWorkerUrl: () => 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs/base/worker/workerMain.js',
      };
    }
    await new Promise((resolve, reject) => {
      if (typeof require === 'function') return resolve();
      const script = document.createElement('script');
      script.async = true;
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js';
      script.onload = resolve;
      script.onerror = () => reject(new Error('require.js failed'));
      document.head.appendChild(script);
    });

    await new Promise((resolve, reject) => {
      if (window.monaco) return resolve();
      const loader = document.createElement('script');
      loader.async = true;
      loader.src = 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs/loader.js';
      loader.onload = () => {
        require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' } });
        require(['vs/editor/editor.main'], () => {
          if (window.monaco) resolve();
          else reject(new Error('monaco main failed'));
        }, reject);
      };
      loader.onerror = () => reject(new Error('monaco loader failed'));
      document.head.appendChild(loader);
    });

    const model = monaco.editor.createModel('', 'javascript');
    const editor = monaco.editor.create(HOST, {
      model,
      automaticLayout: true,
      fontSize: 14,
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      padding: { top: 12 },
      theme: 'vs-dark',
      value: '// Hermes 🤖 AE Hub\n// vscode:// routes live edits through conductor\n',
      language: 'javascript',
    });

    const { surface } = window.HermesSurface || {};
    setLabel('ready');

    editor.onDidChangeModelContent(() => {
      setLabel('dirty');
      try {
        if (surface && typeof surface.dispatch === 'function') {
          surface.dispatch('monaco:change', {
            value: editor.getValue(),
            language: model.getLanguageId(),
          });
        }
      } catch {}
    });

    try {
      if (surface && typeof surface.hydrate === 'function') {
        const doc = await surface.hydrate('ae.hub.monaco');
        if (typeof doc === 'string') editor.setValue(doc);
        else if (doc && doc.value) editor.setValue(String(doc.value));
      }
    } catch {}

    window.HermesMonaco = { editor, model, hydrate: (id) => surface && surface.hydrate(id) };
  }

  bootstrap().catch((err) => {
    setLabel('err:' + (err && err.message ? err.message : 'monaco'));
    console.error('[hermes-monaco]', err);
  });
})();
