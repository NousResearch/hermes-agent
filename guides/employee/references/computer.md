# Your computer

Terminal, file, process and code execution use the configured native Hermes environment. Check its runtime facts before making assumptions.

## Durability and execution

The actual environment is described in your runtime context. `{profile_home}` holds employee knowledge and native profile state. `{workdir}` holds documents, repositories and temporary work. `{guides_root}` holds product-owned guides. These paths are organization conventions, not a filesystem sandbox.

On Railway, only the configured persistent volume survives replacement. Processes do not survive restart; native cron resumes according to native missed-run policy. Save checkpoints in durable files and author recurring work through responsibilities. There is no transparent hosted sleep/wake contract.

Check installed commands and permissions before relying on them. Native terminal and process tools define timeouts, background handles, completion notifications and process ownership. Native `execute_code` defines its supported tool calls and resource limits. Use their actual schemas. Install Hermes dependencies through its package manager; use separate environments for unrelated projects.

Incoming attachments use their native message paths and retention. Keep lasting material under `{workdir}/documents`; repositories under `{workdir}/repos`; disposable work under `{workdir}/tmp`. See `{guides_root}/file-keeping/guide.md`.

All conversations using this profile share its computer. Files, ports and processes can collide; partition parallel work and checkpoint state in files.

## The cloud browser

`browser_exec` runs Python on this computer to control a separate cloud
browser. Its durable workspace profile keeps logins across conversations.
Reuse a session name across calls; different names get separate browsers.
Python variables reset each call, while files and the browser remain.
Browser expiry or insufficient remaining session time can replace the browser;
the profile and workspace files survive, but you may need to navigate again.
Output combines stdout and stderr in a bounded head/tail window. Save large
results in workspace files.

Use `new_tab(url)`, `page_info()`, `fill_input(selector, text)`, `js(expr)`,
and raw `cdp(...)` calls for browser actions. For visual work, call
`print(capture_screenshot())`; the image arrives with the tool result.
Read configured secrets from `os.environ` using the names documented in the service manual. Never print credentials or assume screenshots redact them.

The cloud browser cannot read this computer's paths directly. For a file
input, read the file in Python, base64-encode it, then use `js(...)` to
create a JavaScript `File`, assign it through `DataTransfer` to the input's
`files`, and dispatch its `change` event. Verify the selected file before
submitting. Save extracted data and downloaded bytes under `{workdir}`.
The browser daemon is a tracked background process; native lifecycle management can
restart it, while workspace files and the cloud profile survive.

## When an operation fails

A command error is evidence to investigate. An unknown outcome may have already caused a side effect: inspect files, processes or remote state before retrying. Background processes are process-local; a restart does not prove the work completed. Check the real result and resume from durable checkpoints.
