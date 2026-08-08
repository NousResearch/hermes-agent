# Host Bridge — `remoteUse.nvidia` (full handler)

This is the complete VS Code extension handler that powers the NVIDIA Tiles surface. It shells
out to `nvidia-smi` (and a torch probe) and streams JSON back to the webview.

```ts
const nvidiaCmd = vscode.commands.registerCommand('remoteUse.nvidia', async () => {
  const repo = getHermesRepoPath();
  const hub = `${repo}/templates/surfaces/nvidia-tiles.html`;
  const panel = vscode.window.createWebviewPanel(
    'remoteUseNvidia',
    'Remote Use: NVIDIA Compute Surface',
    vscode.ViewColumn.One,
    { enableScripts: true, retainContextWhenHidden: true,
      localResourceRoots: [vscode.Uri.file(`${repo}/templates/surfaces`)] }
  );
  panel.webview.html = require('fs').readFileSync(hub, 'utf8');
  panel.webview.onDidReceiveMessage(async (message: any) => {
    const action: string = message?.action || '';
    const { exec } = require('child_process');
    const run = (cmd: string) => new Promise<string>((res) =>
      exec(cmd, { maxBuffer: 1024 * 1024 }, (e: any, o: string, err: string) => res((e ? err : o) || '')));

    if (action === 'smi') {
      const out = await run('nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits');
      const parts = out.split(',').map((s: string) => s.trim());
      if (parts.length >= 6) {
        panel.webview.postMessage({ command: 'remoteUse.nvidia', action: 'smi', data: {
          name: parts[0], driver: parts[1],
          memory_used: parseFloat(parts[2]) * 1024 * 1024,
          memory_total: parseFloat(parts[3]) * 1024 * 1024,
          utilization_gpu: parseFloat(parts[4]),
          temperature_gpu: parseFloat(parts[5])
        }});
      }
    } else if (action === 'procs') {
      const out = await run('nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits');
      const procs = out.split('\n').filter(Boolean).map((l: string) => {
        const p = l.split(',').map((s: string) => s.trim());
        return { pid: p[0], name: p[1], used_memory: parseFloat(p[2]) * 1024 * 1024 };
      });
      panel.webview.postMessage({ command: 'remoteUse.nvidia', action: 'procs', data: procs });
    } else if (action === 'torch') {
      let torch = false, version = '';
      try {
        const r = require('child_process').execSync('python -c "import torch;print(torch.__version__)"', { encoding: 'utf8' });
        version = r.trim(); torch = true;
      } catch (e) { /* not installed */ }
      panel.webview.postMessage({ command: 'remoteUse.nvidia', action: 'torch', data: { torch, version } });
    }
  });
});
```

## Webview message contract

Surface → host:
| `action` | payload | host does |
|---|---|---|
| `smi` | — | runs `nvidia-smi --query-gpu=...`, posts `smi` data |
| `procs` | — | runs `nvidia-smi --query-compute-apps=...`, posts `procs` array |
| `torch` | — | `python -c "import torch"`, posts `{torch, version}` |
| `open` | — | `vscode://hermes-agent.remoteUse.nvidia` |

Host → surface:
| `command` | `action` | payload |
|---|---|---|
| `remoteUse.nvidia` | `smi` | `{name, driver, memory_used, memory_total, utilization_gpu, temperature_gpu}` (bytes) |
| `remoteUse.nvidia` | `procs` | `[{pid, name, used_memory}]` (bytes; may be NaN → render `n/a`) |
| `remoteUse.nvidia` | `torch` | `{torch: bool, version: str}` |

## Registration (do not forget either)

`package.json`:
```json
{ "command": "remoteUse.nvidia", "title": "Remote Use: NVIDIA Compute Surface" }
```
`extension.ts` `context.subscriptions.push(...)`:
```ts
nvidiaCmd,
```
