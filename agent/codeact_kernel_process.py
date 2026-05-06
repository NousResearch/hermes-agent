"""Hermes CodeAct kernel subprocess.

Runs as a long-lived child process.  The parent (HermesKernel) communicates
over a Unix domain socket using newline-delimited JSON.

Protocol (parent → kernel):
    {"type": "init",       "namespace_source": "<python src>"}
    {"type": "exec",       "code": "<python src>"}
    {"type": "soft_reset"}
    {"type": "quit"}

Protocol (kernel → parent):
    {"type": "ready"}
    {"type": "tool_call",  "tool": "<name>", "args": {...}}   # mid-exec
    {"type": "exec_result","stdout": "...", "stderr": "...",
                           "last_value": "...", "status": "ok"|"error",
                           "traceback": "..."}

Protocol (parent → kernel, replying to tool_call):
    {"type": "tool_result","result": "..."}
    {"type": "tool_result","result": null, "error": "..."}    # tool error

The _call_tool() function injected into the kernel namespace sends tool_call
messages and blocks waiting for tool_result — this is safe because the parent
enters a receive loop immediately after sending "exec".
"""

from __future__ import annotations

import io
import json
import os
import sys
import traceback


# ---------------------------------------------------------------------------
# Socket I/O helpers
# ---------------------------------------------------------------------------

def _make_conn(sock_path: str):
    """Connect to the UDS socket and return (sock, sock_file)."""
    import socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    sock_file = sock.makefile("rb")
    return sock, sock_file


def _send(sock, msg: dict) -> None:
    sock.sendall((json.dumps(msg, ensure_ascii=False) + "\n").encode())


def _recv(sock_file) -> dict:
    line = sock_file.readline()
    if not line:
        raise EOFError("Parent process disconnected")
    return json.loads(line.strip())


# ---------------------------------------------------------------------------
# Tool IPC bridge (injected into the kernel namespace)
# ---------------------------------------------------------------------------

def _build_call_tool_fn(sock, sock_file):
    """Return the _call_tool function closed over the socket."""

    def _call_tool(tool_name: str, args: dict) -> str:  # noqa: E306
        """Send a tool call request to the parent process and return the result."""
        _send(sock, {"type": "tool_call", "tool": tool_name, "args": args})
        response = _recv(sock_file)
        if response.get("error"):
            raise RuntimeError(
                f"Tool '{tool_name}' returned an error: {response['error']}"
            )
        return response.get("result", "")

    return _call_tool


# ---------------------------------------------------------------------------
# Kernel state
# ---------------------------------------------------------------------------

# The single persistent globals dict — lives for the entire process lifetime.
# Tool stubs + skill functions are injected at init and protected from reset.
_globals: dict = {}


def _eval_last_expr(code: str, ns: dict):
    """If the last statement in code is an expression, eval and return its value."""
    try:
        import ast as _ast
        tree = _ast.parse(code)
        if tree.body and isinstance(tree.body[-1], _ast.Expr):
            return eval(  # noqa: S307
                compile(_ast.Expression(tree.body[-1].value), "<codeact_last>", "eval"),
                ns,
            )
    except Exception:
        pass
    return None


def _do_init(msg: dict, sock, sock_file) -> dict:
    """Exec the namespace source into _globals and register _call_tool."""
    _globals["_call_tool"] = _build_call_tool_fn(sock, sock_file)
    namespace_source = msg.get("namespace_source", "")
    try:
        exec(compile(namespace_source, "<codeact_namespace>", "exec"), _globals)  # noqa: S102
    except Exception:
        return {"type": "error", "traceback": traceback.format_exc()}
    return {"type": "ready"}


def _do_exec(msg: dict) -> dict:
    """Execute user code in the persistent _globals dict."""
    code = msg.get("code", "")
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # Redirect stdout/stderr for the duration of this exec block.
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf

    status = "ok"
    tb = ""
    last_value = None

    try:
        compiled = compile(code, "<codeact>", "exec")
        exec(compiled, _globals)  # noqa: S102
        last_value = _eval_last_expr(code, _globals)
    except Exception:
        status = "error"
        tb = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    result: dict = {
        "type": "exec_result",
        "status": status,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
    }
    if last_value is not None:
        result["last_value"] = repr(last_value)
    if tb:
        result["traceback"] = tb

    return result


def _do_soft_reset(_msg: dict) -> dict:
    """Remove all user-defined names from _globals, preserving protected entries."""
    protected = set(_globals.get("__protected__", []))
    # Always keep dunder keys and _call_tool regardless of __protected__ list.
    to_remove = [
        k for k in list(_globals.keys())
        if k not in protected and not k.startswith("__") and k != "_call_tool"
    ]
    for k in to_remove:
        del _globals[k]
    return {"type": "reset_done", "removed": len(to_remove)}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    sock_path = os.environ.get("HERMES_CODEACT_SOCKET")
    if not sock_path:
        sys.stderr.write("HERMES_CODEACT_SOCKET not set\n")
        sys.exit(1)

    try:
        sock, sock_file = _make_conn(sock_path)
    except Exception as exc:
        sys.stderr.write(f"Failed to connect to kernel socket {sock_path!r}: {exc}\n")
        sys.exit(1)

    while True:
        try:
            msg = _recv(sock_file)
        except (EOFError, json.JSONDecodeError):
            break

        msg_type = msg.get("type", "")

        if msg_type == "init":
            response = _do_init(msg, sock, sock_file)
            _send(sock, response)

        elif msg_type == "exec":
            response = _do_exec(msg)
            _send(sock, response)

        elif msg_type == "soft_reset":
            response = _do_soft_reset(msg)
            _send(sock, response)

        elif msg_type == "quit":
            _send(sock, {"type": "bye"})
            break

        else:
            _send(sock, {"type": "error", "error": f"Unknown message type: {msg_type!r}"})

    sock.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
