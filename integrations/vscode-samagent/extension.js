// SamAgent Local Platform Bridge for VS Code / Cursor / VSCodium
const vscode = require("vscode");
const http = require("http");

function getPlatformUrl() {
  const cfg = vscode.workspace.getConfiguration("samagent");
  return cfg.get("platformUrl") || "http://127.0.0.1:8080";
}

function postJson(urlStr, payload) {
  return new Promise((resolve, reject) => {
    try {
      const url = new URL(urlStr);
      const body = Buffer.from(JSON.stringify(payload || {}), "utf-8");
      const req = http.request(
        {
          hostname: url.hostname,
          port: url.port || 8080,
          path: url.pathname,
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Content-Length": body.length
          }
        },
        (res) => {
          let raw = "";
          res.on("data", (chunk) => (raw += chunk));
          res.on("end", () => {
            try {
              resolve(JSON.parse(raw));
            } catch (e) {
              resolve({ ok: false, raw });
            }
          });
        }
      );
      req.on("error", reject);
      req.write(body);
      req.end();
    } catch (err) {
      reject(err);
    }
  });
}

function activate(context) {
  const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBarItem.text = "$(shield) SamAgent: Ready";
  statusBarItem.tooltip = "Click to run SamAgent L0–L4 Pre-Production Verification";
  statusBarItem.command = "samagent.verifyWorkspace";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  const runVerify = async () => {
    statusBarItem.text = "$(sync~spin) SamAgent: Verifying L0–L4...";
    try {
      const base = getPlatformUrl();
      const res = await postJson(`${base}/api/plugins/samagent/reverify`, {});
      const gate = res && res.pre_prod_gate;
      if (gate && gate.ready_for_production) {
        statusBarItem.text = "$(pass-filled) SamAgent: Pre-Prod Ready (L0–L4 PASS)";
        vscode.window.showInformationMessage("SamAgent Pre-Production Gate PASSED (L0–L4 + OWASP Security). Ready for deployment.");
      } else {
        statusBarItem.text = "$(error) SamAgent: Pre-Prod Gate Blocked";
        const blockers = (gate && gate.blockers ? gate.blockers.join("; ") : "Verification failed");
        vscode.window.showWarningMessage(`SamAgent Pre-Production Gate Blocked: ${blockers}`);
      }
    } catch (err) {
      statusBarItem.text = "$(warning) SamAgent Platform Offline";
      vscode.window.showErrorMessage(`Cannot reach local SamAgent Platform at ${getPlatformUrl()}`);
    }
  };

  context.subscriptions.push(
    vscode.commands.registerCommand("samagent.verifyWorkspace", runVerify),
    vscode.commands.registerCommand("samagent.checkPreProdGate", runVerify),
    vscode.commands.registerCommand("samagent.openMissionControl", () => {
      vscode.env.openExternal(vscode.Uri.parse(getPlatformUrl()));
    })
  );

  // Auto-verify on file save in VS Code so developer sees instant L0-L4 + OWASP feedback before deploying
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument((doc) => {
      const cfg = vscode.workspace.getConfiguration("samagent");
      if (cfg.get("verifyOnSave") && doc.uri.fsPath && !doc.uri.fsPath.includes(".git")) {
        runVerify();
      }
    })
  );

  // Embedded Mission Control Webview Sidebar
  const provider = {
    resolveWebviewView(webviewView) {
      webviewView.webview.options = { enableScripts: true };
      const url = getPlatformUrl();
      webviewView.webview.html = `<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;height:100vh;overflow:hidden;background:#f8fafc;">
  <iframe src="${url}" style="border:none;width:100%;height:100vh;"></iframe>
</body>
</html>`;
    }
  };
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("samagent.missionControlView", provider)
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
