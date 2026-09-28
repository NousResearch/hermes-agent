"""Shared fixture HERMES_HOME + invariant checks for the entrypoint-parity suite.

One fixture home carries every feature that has historically been wired into
some entrypoints and forgotten in others (issue class C19): a shell hook and a
Python-plugin hook on ``pre_llm_call`` (both write a marker AND inject a
context canary), an ``AGENTS.md`` context file in the working directory, a
skill, a memory entry, a stdio MCP server (issue class C15) whose tool returns a
canary, the custom provider pointed at the recording fake, and a toolset
restriction (``agent.disabled_toolsets``).

Every entrypoint driver runs ONE turn against the scripted fake provider
(turn 1: call the MCP tool; turn 2: answer) and hands the recorded requests to
:func:`collect_observation`; the test then asserts the same invariants for
every entrypoint, so any red cell is a wiring-parity bug.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_MCP_SERVER = Path(__file__).with_name("fixture_mcp_server.py")

MCP_SERVER_NAME = "parity"
MCP_TOOL_NAME = "mcp__parity__parity_canary"
FINAL_ANSWER = "PARITY-TURN-COMPLETE"
PLUGIN_NAME = "parity-plugin"
# Always-available (no credential/check_fn gate) toolset the fixture disables.
DISABLED_TOOLSET = "todo"

_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_ACCESS_KEY")
_PASSTHROUGH_ENV = frozenset({
    "PATH", "LANG", "LANGUAGE", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ",
    "SYSTEMROOT", "SystemRoot", "COMSPEC", "PATHEXT", "WINDIR", "TEMP", "TMP",
})


TURN_TIMEOUT = 240.0


@dataclass
class DriveResult:
    """What an entrypoint driver reports back (see ``_drive_cli`` for the contract)."""

    final_text: str | None
    toolset: str  # the platform toolset this surface documents (a toolsets.py key)
    cwd_channel: str = "launch dir"
    graceful_exit: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParityHome:
    """Filesystem layout + canaries of one fixture home."""

    root: Path
    home: Path
    hermes_home: Path
    project: Path
    markers: Path
    pid_log: Path
    tag: str
    canaries: dict[str, str] = field(default_factory=dict)

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Hermetic env for a subprocess Hermes: fake HOME, no real credentials."""
        import pwd  # POSIX-only; the suite is Linux-gated

        # Refuse only a home the real install would read as live state (its root or a profile).
        # A tmp_path under ``~/.hermes/cache/scratch`` (TMPDIR when Hermes itself runs the suite)
        # is fine: the child's HOME is the fixture home, so its ``~/.hermes`` never resolves there.
        real_root = Path(pwd.getpwuid(os.getuid()).pw_dir, ".hermes").resolve()
        fixture = self.hermes_home.resolve()
        assert fixture != real_root and fixture.parent != real_root / "profiles", (
            f"fixture HERMES_HOME {self.hermes_home} is the real install's live home")
        assert fixture == (self.home / ".hermes").resolve(), (
            f"fixture HERMES_HOME {self.hermes_home} is not <fixture HOME>/.hermes")
        # Allowlist, not denylist: the runner may itself be a Hermes process whose
        # TERMINAL_CWD / HERMES_* / credential env would silently reroute the child.
        env = {
            k: v for k, v in os.environ.items()
            if (k in _PASSTHROUGH_ENV or k.startswith("LC_")) and not k.endswith(_SECRET_ENV_SUFFIXES)
        }
        env.update({
            "HOME": str(self.home),
            "HERMES_HOME": str(self.hermes_home),
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
            # Orphan-scan tag: every process in the spawned tree inherits it.
            "PARITY_TREE_TAG": self.tag,
            # The child's HOME is the fixture home, so its ``~/.hermes/state.db`` IS
            # the tmp HERMES_HOME's db; under a pytest ancestor the live-DB guard
            # (hermes_state_guard) would refuse it. This is the guard's documented
            # child-process escape hatch; the path is tmp_path by construction.
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
        })
        env.update(extra or {})
        return env

    def update_config(self, mutate: Callable[[dict], None]) -> None:
        """Apply ``mutate`` to config.yaml (for surfaces with a documented config-only channel)."""
        path = self.hermes_home / "config.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        mutate(cfg)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    def pin_terminal_cwd(self) -> None:
        """Documented cwd channel for daemon surfaces (gateway, api_server): ``terminal.cwd``."""
        self.update_config(lambda cfg: cfg.setdefault("terminal", {}).__setitem__("cwd", str(self.project)))


def build_parity_home(root: Path, base_url: str, *, grandchild: bool = True,
                      death_tool: bool = False, mcp_timeout: int = 60) -> ParityHome:
    """Write the full fixture home under ``root`` (a tmp_path)."""
    home = root / "home"
    hermes_home = home / ".hermes"
    project = root / "project"
    markers = root / "markers"
    for d in (hermes_home, project, markers):
        d.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex
    c = {name: f"{name.upper()}-{uuid.uuid4().hex[:12]}" for name in (
        "context", "skill", "memory", "shell_hook", "plugin_hook", "mcp")}
    ph = ParityHome(root=root, home=home, hermes_home=hermes_home, project=project,
                    markers=markers, pid_log=root / "mcp_pids.log", tag=tag, canaries=c)

    write_hermes_home(hermes_home, base_url)
    cfg = yaml.safe_load((hermes_home / "config.yaml").read_text(encoding="utf-8"))
    cfg["agent"]["disabled_toolsets"] = [DISABLED_TOOLSET]
    cfg["hooks"] = {"pre_llm_call": [{
        "command": f"{sys.executable} {hermes_home / 'agent-hooks' / 'shell_hook.py'}",
        "timeout": 30,
    }]}
    cfg["hooks_auto_accept"] = True
    cfg["plugins"] = {"enabled": [PLUGIN_NAME]}
    cfg["mcp_servers"] = {MCP_SERVER_NAME: {
        "command": sys.executable,
        "args": [str(FIXTURE_MCP_SERVER)],
        "env": {
            "PARITY_MCP_CANARY": c["mcp"],
            "PARITY_MCP_PID_LOG": str(ph.pid_log),
            "PARITY_MCP_SPAWN_GRANDCHILD": "1" if grandchild else "0",
            "PARITY_MCP_DEATH_TOOL": "1" if death_tool else "0",
            "PARITY_TREE_TAG": tag,
            # Only the MCP server and its descendants carry this one.
            "PARITY_MCP_TREE_TAG": tag,
            "PYTHONPATH": str(REPO_ROOT),
        },
        "connect_timeout": 60,
        "timeout": mcp_timeout,
    }}
    # Interactive surfaces wait only ~1.5 s (mcp_discovery_timeout) for MCP discovery
    # before the first agent build, BY DESIGN (a slow server must not block the
    # shell; late tools arrive via refresh). Under a loaded box the fixture server's
    # import alone can exceed that, so pin the documented knob high: the join returns
    # the instant discovery finishes, and the first turn is deterministically complete.
    cfg["mcp_discovery_timeout"] = 120
    cfg["mcp_single_query_discovery_timeout"] = 120
    # Keep turns hermetic and short: no title/aux model chatter decides anything here.
    cfg.setdefault("display", {})["compact"] = True
    cfg["updates"] = {"check": False}  # offline: no GitHub round-trip or git lazy fetch
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    hooks_dir = hermes_home / "agent-hooks"
    hooks_dir.mkdir()
    (hooks_dir / "shell_hook.py").write_text(
        "import json, os, sys\n"
        "payload = json.load(sys.stdin)\n"
        f"with open({str(markers / 'shell_hook.log')!r}, 'a') as fh:\n"
        "    fh.write(json.dumps({'event': payload.get('hook_event_name'),\n"
        "                         'session_id': payload.get('session_id')}) + '\\n')\n"
        f"print(json.dumps({{'context': {c['shell_hook']!r}}}))\n",
        encoding="utf-8",
    )

    plugin_dir = hermes_home / "plugins" / PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {PLUGIN_NAME}\nversion: 1.0.0\ndescription: parity suite marker plugin\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "import json\n"
        "def register(ctx):\n"
        "    def _pre_llm_call(**kwargs):\n"
        f"        with open({str(markers / 'plugin_hook.log')!r}, 'a') as fh:\n"
        "            fh.write(json.dumps({'platform': kwargs.get('platform'),\n"
        "                                 'session_id': kwargs.get('session_id')}) + '\\n')\n"
        f"        return {{'context': {c['plugin_hook']!r}}}\n"
        "    ctx.register_hook('pre_llm_call', _pre_llm_call)\n",
        encoding="utf-8",
    )

    skill_dir = hermes_home / "skills" / "parity-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: parity-skill\n"
        f"description: Use when checking entrypoint parity {c['skill']}.\n---\n\n"
        "# Parity skill\n\nNothing to do.\n",
        encoding="utf-8",
    )

    mem_dir = hermes_home / "memories"
    mem_dir.mkdir()
    (mem_dir / "MEMORY.md").write_text(f"The parity memory canary is {c['memory']}.", encoding="utf-8")

    (project / "AGENTS.md").write_text(
        f"# Project rules\n\nThe parity context canary is {c['context']}.\n", encoding="utf-8")
    return ph


# Scripted provider -----------------------------------------------------------


def parity_responder(nonce: str, tool: str = MCP_TOOL_NAME) -> Callable[[dict[str, Any]], Any]:
    """Stateless script: call the MCP tool until a tool result exists, then answer.

    Stateless so a retried request (or a second concurrent entrypoint request)
    cannot desynchronise the script.
    """

    def respond(record: dict[str, Any]):
        body = record["body"]
        if any(m.get("role") == "tool" for m in body.get("messages") or []):
            return Text(FINAL_ANSWER)
        args = {"nonce": nonce}
        if tool in _tool_names(body) or "tool_call" not in _tool_names(body):
            return ToolCall(tool, args)
        # Tool Search active: MCP tools sit in the deferred catalog behind the bridge.
        return ToolCall("tool_call", {"calls": [{"name": tool, "arguments": args}]})

    return respond


def start_provider(nonce: str, tool: str = MCP_TOOL_NAME) -> FakeLLMServer:
    srv = FakeLLMServer(parity_responder(nonce, tool))
    srv.start()
    return srv


# Observation + invariants ----------------------------------------------------


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _tool_names(body: dict[str, Any]) -> set[str]:
    return {(t.get("function") or {}).get("name") or t.get("name") for t in body.get("tools") or []}


_CATALOG_LINE = re.compile(r"^- ([A-Za-z0-9_.:-]+): ", re.M)


def offered_tool_names(body: dict[str, Any]) -> set[str]:
    """Direct tool schemas plus the Tool Search deferred catalog (names the model may invoke)."""
    names = _tool_names(body)
    for t in body.get("tools") or []:
        fn = t.get("function") or {}
        if fn.get("name") == "tool_search":
            names |= set(_CATALOG_LINE.findall(fn.get("description") or ""))
    return names


@dataclass
class Observation:
    entrypoint: str
    system: str
    first_user: str
    tool_names: set[str]
    tool_results: list[str]
    main_requests: int
    shell_hook_events: list[dict]
    plugin_hook_events: list[dict]
    final_text: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_observation(entrypoint: str, ph: ParityHome, srv: FakeLLMServer,
                        final_text: str | None = None) -> Observation:
    mains = srv.main_requests()
    assert mains, f"{entrypoint}: no main-turn request reached the provider"
    first = mains[0]
    msgs = first.get("messages") or []
    system = "\n".join(_text(m.get("content")) for m in msgs if m.get("role") == "system")
    users = [_text(m.get("content")) for m in msgs if m.get("role") == "user"]
    tool_results = [
        _text(m.get("content")) for body in mains for m in body.get("messages") or []
        if m.get("role") == "tool"
    ]
    return Observation(
        entrypoint=entrypoint, system=system, first_user=users[-1] if users else "",
        tool_names=offered_tool_names(first), tool_results=tool_results, main_requests=len(mains),
        shell_hook_events=_read_jsonl(ph.markers / "shell_hook.log"),
        plugin_hook_events=_read_jsonl(ph.markers / "plugin_hook.log"),
        final_text=final_text,
    )


def disabled_tool_names() -> set[str]:
    from toolsets import resolve_toolset

    return set(resolve_toolset(DISABLED_TOOLSET))


# Agent features every surface must carry when its documented toolset includes them.
_FEATURE_TOOLS = ("terminal", "read_file", "write_file", "patch", "search_files", "memory",
                  "skills_list", "skill_view")


def required_tool_names(toolset: str) -> set[str]:
    """The core feature tools the surface's documented toolset promises, plus the MCP tool."""
    from toolsets import resolve_toolset

    promised = set(resolve_toolset(toolset))
    return {t for t in _FEATURE_TOOLS if t in promised} | {MCP_TOOL_NAME}


def check_invariants(obs: Observation, ph: ParityHome, *, toolset: str = "hermes-cli",
                     context_file: bool = True) -> dict[str, bool]:
    """Evaluate every parity invariant; returns {cell: ok} (all must be True)."""
    c = ph.canaries
    cells = {
        "context_file": c["context"] in obs.system,
        "skill_index": c["skill"] in obs.system,
        "memory": c["memory"] in obs.system,
        "shell_hook_fired": bool(obs.shell_hook_events),
        "shell_hook_injected": c["shell_hook"] in obs.first_user,
        "plugin_hook_fired": bool(obs.plugin_hook_events),
        "plugin_hook_injected": c["plugin_hook"] in obs.first_user,
        "mcp_tool_offered": MCP_TOOL_NAME in obs.tool_names,
        "mcp_call_returned_canary": any(c["mcp"] in r for r in obs.tool_results),
        "toolset_restriction": not (obs.tool_names & disabled_tool_names()),
        "feature_tools": required_tool_names(toolset) <= obs.tool_names,
        "turn_completed": obs.main_requests >= 2,
    }
    if not context_file:
        cells.pop("context_file")
    return cells


# Process-tree hygiene --------------------------------------------------------


def tagged_pids(tag: str, var: str = "PARITY_TREE_TAG") -> set[int]:
    """Live PIDs whose environment carries ``<var>=<tag>`` (reparented orphans included)."""
    needle = f"{var}={tag}".encode()
    found: set[int] = set()
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as fh:
                env = fh.read()
            with open(f"/proc/{entry}/stat", "rb") as fh:
                state = fh.read().rsplit(b")", 1)[1].split()[0]
        except OSError:
            continue
        if state != b"Z" and needle in env.split(b"\0"):
            found.add(int(entry))
    return found


def mcp_pids(ph: ParityHome) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    if ph.pid_log.exists():
        for line in ph.pid_log.read_text(encoding="utf-8").splitlines():
            kind, pid = line.split()
            out.setdefault(kind, []).append(int(pid))
    return out


def wait_until(pred: Callable[[], Any], timeout: float, what: str, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while True:
        value = pred()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        time.sleep(interval)


def wait_no_orphans(ph: ParityHome, timeout: float = 30.0, *, mcp_only: bool = True) -> set[int]:
    """Poll until no MCP-tree process (or, ``mcp_only=False``, no tagged process at all)
    survives; returns the survivors (empty = clean)."""
    var = "PARITY_MCP_TREE_TAG" if mcp_only else "PARITY_TREE_TAG"
    deadline = time.monotonic() + timeout
    survivors = tagged_pids(ph.tag, var)
    while survivors and time.monotonic() < deadline:
        time.sleep(0.1)
        survivors = tagged_pids(ph.tag, var)
    return survivors


def describe_pids(pids: Iterable[int]) -> list[str]:
    out = []
    for pid in sorted(pids):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode(errors="replace")
            out.append(f"{pid}: {cmd[:160]}")
        except OSError:
            out.append(f"{pid}: <gone>")
    return out


def kill_tagged(ph: ParityHome) -> None:
    """Test cleanup: SIGKILL anything this fixture spawned that is still alive."""
    for pid in tagged_pids(ph.tag):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def hermes_argv(*args: str) -> list[str]:
    return [sys.executable, "-m", "hermes_cli.main", *args]


def terminate(proc: subprocess.Popen, timeout: float = 30.0) -> int | None:
    """Graceful SIGTERM (the normal stop), SIGKILL only if it overstays."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    return proc.returncode


def format_cells(results: Iterable[tuple[str, dict[str, bool]]]) -> str:
    rows = list(results)
    cols = sorted({k for _, cells in rows for k in cells})
    head = "| entrypoint | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    body = [
        f"| {ep} | " + " | ".join(("✅" if cells.get(k) else ("—" if k not in cells else "❌")) for k in cols) + " |"
        for ep, cells in rows
    ]
    return "\n".join([head, sep, *body])
