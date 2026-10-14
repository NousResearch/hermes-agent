"""Generated logins remain pending until finalized and never expose their password."""
import json
import subprocess

import pytest


def test_generated_lifecycle_scopes_commands_and_rejects_reuse(monkeypatch, caplog):
    from agent.vault_backends.onepassword import OnePasswordLoginBackend
    from agent.vault_backends import onepassword

    backend = OnePasswordLoginBackend()
    backend._service_token = "service-token"
    monkeypatch.setattr(backend, "_op", lambda: "/op")
    calls = []
    items = {}
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1:3] == ["vault", "list"]:
            result = [{"id": "vault-id"}]
        elif argv[1:3] == ["item", "create"]:
            items["a" * 26] = argv[argv.index("--title") + 1]
            result = {"id": "a" * 26, "fields": [{"value": "secret-canary"}]}
        elif argv[1:3] == ["item", "list"]:
            result = [{"id": key, "title": value, "urls": [{"href": "https://example.test"}]} for key, value in items.items()]
        else:
            result = {}
        return subprocess.CompletedProcess(argv, 0, json.dumps(result), "")
    monkeypatch.setattr(onepassword, "run_cli", run)
    handle = backend.create_generated_login("https://example.test", "Example")
    assert "secret-canary" not in handle
    pending_title = items["a" * 26]
    assert pending_title.startswith("[Pending signup ")
    items["a" * 26] = "Manually finalized"
    with pytest.raises(ValueError):
        backend.discard_generated_login(handle)
    items["a" * 26] = pending_title
    backend.commit_generated_login(handle, "Finished")
    with pytest.raises(ValueError):
        backend.discard_generated_login(handle)
    with pytest.raises(ValueError):
        backend.commit_generated_login("op:unknown", "Finished")
    handle = backend.create_generated_login("https://example.test", "Example")
    backend.discard_generated_login(handle)
    with pytest.raises(ValueError):
        backend.discard_generated_login(handle)
    mutations = [a for a, _ in calls if a[1:3] in (["item", "create"], ["item", "edit"], ["item", "delete"])]
    assert all(a[a.index("--vault") + 1] == "vault-id" for a in mutations)
    assert any("--generate-password=letters,digits,symbols,32" in a for a in mutations)
    assert "secret-canary" not in repr(calls)
    assert all(k["stdin"] == subprocess.DEVNULL for _, k in calls)

    def multiple_vaults(argv, **kwargs):
        assert argv[1:3] == ["vault", "list"]  # No write may follow ambiguous scope.
        return subprocess.CompletedProcess(argv, 0, '[{"id":"one"},{"id":"two"}]', "")
    monkeypatch.setattr(onepassword, "run_cli", multiple_vaults)
    with pytest.raises(RuntimeError, match="creation failed"):
        backend.create_generated_login("https://example.test", "Example")
    assert not backend._generated

    import traceback
    for failure in (subprocess.CompletedProcess([], 1, "secret-canary", "secret-canary"),
                    subprocess.TimeoutExpired(["op"], 30, output="secret-canary")):
        def fail(argv, **kwargs):
            if isinstance(failure, Exception):
                raise failure
            return failure
        monkeypatch.setattr(onepassword, "run_cli", fail)
        with pytest.raises(RuntimeError) as exc:
            backend.create_generated_login("https://example.test", "Example")
        assert "secret-canary" not in "".join(traceback.format_exception(exc.value))
    assert "secret-canary" not in caplog.text
    before = len(calls)
    for origin, label in [("file:///tmp", "Example"), ("https://example.test/path", "Example"),
                          ("https://example.test", "--reveal"), ("https://example.test", "bad\nlabel")]:
        with pytest.raises(ValueError):
            backend.create_generated_login(origin, label)
    assert len(calls) == before
    monkeypatch.setattr(onepassword, "run_cli", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 1, "", "secret-canary"))
    with pytest.raises(RuntimeError) as caught:
        backend.create_generated_login("https://example.test", "Example")
    assert "secret-canary" not in str(caught.value)


@pytest.fixture
def signup_backend(monkeypatch, tmp_path):
    """Run the real backend and CLI transport against a local op protocol fixture."""
    import sys
    from agent.vault_backends.onepassword import OnePasswordLoginBackend
    from agent.vault_backends import base
    from tools import browser_vault_tool as tool
    from agent import redact

    executable = tmp_path / "op"
    executable.write_text("#!" + sys.executable + "\n" + r'''import json, pathlib, sys
args = sys.argv[1:]
state = pathlib.Path(__file__).with_suffix('.json')
items = json.loads(state.read_text()) if state.exists() else {}
secret = 'signup-secret-canary'
assert secret not in repr(args)
if args[:2] == ['vault', 'list']:
    result = [{'id': 'vault-id'}]
elif args[:2] == ['item', 'create']:
    assert args[args.index('--vault') + 1] == 'vault-id'
    assert '--generate-password=letters,digits,symbols,32' in args
    item = {'id': 'a' * 26, 'title': args[args.index('--title') + 1],
            'urls': [{'href': args[args.index('--url') + 1]}]}
    items[item['id']] = item
    result = dict(item, fields=[{'value': secret}])
elif args[:2] == ['item', 'list']:
    result = list(items.values())
elif args[:2] == ['item', 'get']:
    assert args[args.index('--vault') + 1] == 'vault-id'
    print(secret)
    sys.exit(0)
elif args[:2] == ['item', 'edit']:
    assert args[args.index('--vault') + 1] == 'vault-id'
    items[args[2]]['title'] = args[args.index('--title') + 1]
    result = {'fields': [{'value': secret}]}
elif args[:2] == ['item', 'delete']:
    assert args[args.index('--vault') + 1] == 'vault-id'
    del items[args[2]]
    result = {}
else:
    raise RuntimeError('Unexpected command')
state.write_text(json.dumps(items))
print(json.dumps(result))
''')
    executable.chmod(0o700)
    backend = OnePasswordLoginBackend({"binary_path": str(executable)})
    backend._service_token = "test-service-token"
    monkeypatch.setattr(base, "enabled_backends", lambda: [backend])
    monkeypatch.setattr(tool, "_signup_pending", {})
    yield backend
    redact.clear_vault_redaction_values()


@pytest.fixture
def signup_page(monkeypatch, tmp_path):
    """Execute production expressions in Chromium over a real CDP websocket."""
    import functools
    import http.server
    import shutil
    import threading
    import time
    import urllib.request
    from websockets.sync.client import connect
    from tools import browser_vault_tool as tool

    chrome = shutil.which("chromium") or shutil.which("google-chrome")
    if not chrome:
        from pathlib import Path
        chrome = "/opt/google/chrome/chrome" if Path("/opt/google/chrome/chrome").exists() else None
    if not chrome:
        pytest.skip("Chromium is required for live signup CDP coverage")
    (tmp_path / "index.html").write_text('''<form>
      <input id="current" type="password" autocomplete="current-password" value="untouched">
      <input id="new" type="password" autocomplete="new-password" data-hermes-vault-slot="owned-by-page">
      <input id="confirm" type="password" aria-label="Confirm password">
      <input id="ambiguous" type="password">
    </form>''')
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    profile = tmp_path / "chrome-profile"
    proc = subprocess.Popen([chrome, "--headless", "--no-sandbox", "--disable-gpu",
                             "--remote-debugging-port=0", "--user-data-dir=" + str(profile), "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ws = None
    try:
        port_file = profile / "DevToolsActivePort"
        deadline = time.monotonic() + 15
        while not port_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        port = port_file.read_text().splitlines()[0]
        with urllib.request.urlopen("http://127.0.0.1:" + port + "/json/list") as response:
            targets = json.load(response)
        ws = connect(next(t['webSocketDebuggerUrl'] for t in targets if t['type'] == 'page'))
        serial = 0
        def cdp(method, params):
            nonlocal serial
            serial += 1
            ws.send(json.dumps({"id": serial, "method": method, "params": params}))
            while True:
                result = json.loads(ws.recv(timeout=10))
                if result.get("id") == serial:
                    return result["result"]
        class Supervisor:
            def evaluate_runtime(self, expression):
                result = cdp("Runtime.evaluate", {"expression": expression, "returnByValue": True})
                if "exceptionDetails" in result:
                    return {"ok": False, "error": "page evaluation failed"}
                return {"ok": True, "result": result["result"].get("value")}
        supervisor = Supervisor()
        monkeypatch.setattr(tool, "_ensure_supervisor", lambda task: supervisor)
        def forbidden(*args):
            pytest.fail("Signup must not use a CLI evaluation fallback")
        monkeypatch.setattr(tool, "_eval_js", forbidden)
        origin = "http://127.0.0.1:" + str(server.server_port)
        cdp("Page.navigate", {"url": origin + "/index.html"})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if supervisor.evaluate_runtime("!!document.getElementById('confirm')")["result"]:
                break
            time.sleep(0.05)
        yield supervisor, origin
    finally:
        if ws:
            ws.close()
        proc.terminate()
        proc.wait(timeout=10)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_signup_browser_lifecycle(signup_backend, signup_page, monkeypatch, caplog):
    from tools import browser_vault_tool as tool
    from agent.redact import redact_sensitive_text
    from agent.vault_backends import base
    from tools.registry import registry
    supervisor, origin = signup_page
    evaluate = supervisor.evaluate_runtime
    created = json.loads(tool.create_signup_login("Example", "task"))
    handle = created["handle"]
    assert created["origin"] == origin
    assert not handle.startswith("op:")
    for operation in (lambda task: tool.fill_signup_password(handle, task),
                      lambda task: tool.commit_signup_login(handle, "Finished", task),
                      lambda task: tool.discard_signup_login(handle, task)):
        assert not json.loads(operation("other-task"))["success"]
    assert len(tool._signup_pending) == 1
    original = supervisor.evaluate_runtime
    def checked(expression):
        if "signup-secret-canary" in expression:
            assert "signup-secret-canary" not in redact_sensitive_text("signup-secret-canary")
        return original(expression)
    monkeypatch.setattr(supervisor, "evaluate_runtime", checked)
    result = tool.fill_signup_password(handle, "task")
    assert json.loads(result)["filled_fields"] == 2
    assert evaluate("document.getElementById('current').value")["result"] == "untouched"
    assert evaluate("document.getElementById('ambiguous').value")["result"] == ""
    assert evaluate("document.getElementById('new').value === document.getElementById('confirm').value")["result"]
    assert evaluate("document.getElementById('new').getAttribute('data-hermes-vault-slot')")["result"] == "owned-by-page"
    assert not evaluate("Array.from(document.querySelectorAll('input')).some(e => e.getAttributeNames().some(a => a.startsWith('data-hermes-signup-')))")["result"]
    assert json.loads(tool.commit_signup_login(handle, "Finished", "task"))["success"]
    assert not tool._signup_pending and not signup_backend._generated
    assert not json.loads(tool.discard_signup_login(handle, "task"))["success"]
    second = json.loads(tool.create_signup_login("Example", "task"))["handle"]
    assert json.loads(tool.discard_signup_login(second, "task"))["success"]
    assert not json.loads(tool.fill_signup_password(second, "task"))["success"]
    assert not tool._signup_pending and not signup_backend._generated
    monkeypatch.setattr(tool, "_MAX_SIGNUP_PENDING", 0)
    assert not json.loads(tool.create_signup_login("Example", "task"))["success"]
    monkeypatch.setattr(tool, "_MAX_SIGNUP_PENDING", 128)
    monkeypatch.setattr(base, "enabled_backends", lambda: [object()])
    assert not json.loads(tool.create_signup_login("Example", "task"))["success"]
    assert "signup-secret-canary" not in result + caplog.text
    for name in ("create_signup_login", "fill_signup_password", "commit_signup_login", "discard_signup_login"):
        assert registry.get_schema(name) is None


def test_signup_refuses_changed_page_and_cleans_locators(signup_backend, signup_page, monkeypatch, caplog):
    from tools import browser_vault_tool as tool
    from agent.vault_signup import build_signup_fill_js, build_signup_inspection_js, classify_signup_controls
    supervisor, origin = signup_page
    handle = json.loads(tool.create_signup_login("Example", "task"))["handle"]
    evaluate = supervisor.evaluate_runtime
    # Change the target after inspection, before the secret-bearing evaluation.
    resolve = signup_backend.resolve_password
    def changed(handle):
        evaluate("document.getElementById('new').autocomplete = 'current-password'")
        return resolve(handle)
    monkeypatch.setattr(signup_backend, "resolve_password", changed)
    assert not json.loads(tool.fill_signup_password(handle, "task"))["success"]
    assert evaluate("document.getElementById('new').value")["result"] == ""
    assert evaluate("document.getElementById('confirm').value")["result"] == ""
    assert not evaluate("Array.from(document.querySelectorAll('input')).some(e => e.getAttributeNames().some(a => a.startsWith('data-hermes-signup-')))")["result"]
    evaluate("document.getElementById('new').autocomplete = 'new-password'")
    inspected = json.loads(evaluate(build_signup_inspection_js("abc"))["result"])
    controls = classify_signup_controls(inspected)
    refused = evaluate(build_signup_fill_js(controls, "signup-secret-canary", "https://other.test", "abc"))
    assert json.loads(refused["result"])["filled"] == 0
    assert evaluate("document.getElementById('new').value")["result"] == ""
    def failed(handle):
        raise RuntimeError("signup-secret-canary")
    monkeypatch.setattr(signup_backend, "resolve_password", failed)
    result = tool.fill_signup_password(handle, "task")
    assert not json.loads(result)["success"]
    assert "signup-secret-canary" not in result + caplog.text
    monkeypatch.setattr(signup_backend, "resolve_password", resolve)
    original_evaluate = supervisor.evaluate_runtime
    def cdp_failure(expression):
        if "signup-secret-canary" in expression:
            raise RuntimeError("signup-secret-canary")
        return original_evaluate(expression)
    monkeypatch.setattr(supervisor, "evaluate_runtime", cdp_failure)
    result = tool.fill_signup_password(handle, "task")
    assert not json.loads(result)["success"]
    assert "signup-secret-canary" not in result + caplog.text
    monkeypatch.setattr(tool, "_ensure_supervisor", lambda task: None)
    assert not json.loads(tool.fill_signup_password(handle, "task"))["success"]
    assert not json.loads(tool.create_signup_login("Example", "task"))["success"]
    # Finalizing a known handle needs no page: registration may redirect or close it.
    assert json.loads(tool.discard_signup_login(handle, "task"))["success"]
