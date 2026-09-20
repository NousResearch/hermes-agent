"""A zero-argument ``oauth_flow`` companion runs in a child bound to the owning profile, because its
launcher thread reads HERMES_HOME and .env after the request returns."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

# Honcho's loopback hook waits up to 300s for consent; leave room for exchange and persistence.
CHILD_LIFETIME = 660.0
# Deployment transport/browser settings shared by every owner; an owner's .env may override them.
_DEPLOYMENT_ENV = frozenset({
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "TMP", "TEMP",
    "DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS",
})


def _child_env(home: Path) -> dict:
    """Deployment values plus this owner's secrets and terminal policy; never the ambient shell."""
    from agent.secret_scope import _GLOBAL_ENV_EXACT, build_profile_secret_scope
    from hermes_cli.plugin_installation import is_launch_home
    from tools.code_execution_env import _WINDOWS_ESSENTIAL_ENV_VARS
    from tools.terminal_scope import enforce_no_refusal, get_terminal_scope
    from tui_gateway.launch_profile_policy import launch_secret_scope

    home = Path(home).resolve()
    env = {key: value for key, value in os.environ.items()
           if key in _GLOBAL_ENV_EXACT or key in _DEPLOYMENT_ENV
           or (os.name == "nt" and key.upper() in _WINDOWS_ESSENTIAL_ENV_VARS)}
    env.update(launch_secret_scope(home) if is_launch_home(home) else build_profile_secret_scope(home))
    enforce_no_refusal()
    env = {key: value for key, value in env.items() if not key.startswith("TERMINAL_")}
    env.update(get_terminal_scope() or {})  # management_scope installed the owner's policy
    env.pop("HERMES_PROFILE", None)  # The explicit home, never the launch profile's label.
    env["HERMES_HOME"] = str(home)
    env.setdefault("PYTHONUTF8", "1")
    return env


class OAuthChild:
    def __init__(self, home: Path, directory: Path):
        self._directory = str(directory)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.memory_oauth_process"],
            cwd=str(Path(__file__).resolve().parents[1]), env=_child_env(home),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self._timer = threading.Timer(CHILD_LIFETIME, self.close)
        self._timer.daemon = True
        self._timer.start()

    def request(self, *, start: bool):
        wire = json.dumps({"operation": "start" if start else "status", "directory": self._directory}).encode() + b"\n"
        result: queue.Queue = queue.Queue(maxsize=1)

        def exchange():
            try:
                self.process.stdin.write(wire)
                self.process.stdin.flush()
                result.put(json.loads(self.process.stdout.readline()))
            except Exception:
                result.put(None)

        threading.Thread(target=exchange, name="memory-oauth-rpc", daemon=True).start()
        try:
            response = result.get(timeout=15.0)
        except queue.Empty:
            response = None
        if not isinstance(response, dict):
            self.close()
            raise RuntimeError("OAuth child request failed")
        if response.get("ok") is not True:  # the hook raised; the child and its flow are still running
            raise RuntimeError("OAuth companion hook failed")
        return response.get("payload")

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def close(self):
        self._timer.cancel()
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()


def main():
    from plugins.memory.contract import load_oauth_companion
    from plugins.package_generation import capture_package

    # Reserve the pipe, then point fd 1/2 at devnull so no plugin output reaches the RPC channel.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    flow = None
    while line := sys.stdin.buffer.readline():
        try:
            request = json.loads(line)
            if flow is None:
                flow = load_oauth_companion(capture_package(Path(request["directory"])))
            payload = flow.start_loopback_flow_background() if request["operation"] == "start" else flow.get_flow_status()
            reply = {"ok": True, "payload": payload}
        except Exception:
            reply = {"ok": False}
        output.write(json.dumps(reply) + "\n")
    os._exit(0)  # EOF means the host is gone; provider threads need not be daemon threads.


if __name__ == "__main__":
    main()
