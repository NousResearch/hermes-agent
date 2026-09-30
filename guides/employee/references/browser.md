# Browser Use Cloud


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
