"""Dangerous terminal commands need a human's approval — through the REAL terminal tool path.

Two surfaces, both real processes, the model scripted by ``FakeLLMServer``:

* CLI: ``hermes chat -q`` (single-query, nobody present to approve; ``approvals.mode: manual`` so
  no guardian LLM decides). The fake model calls ``terminal`` with a table of rm -rf spellings meant
  to slip past the detector (quoting, backslash, ``env``, sub-shells, ``$(...)``, absolute path,
  ``find -delete``, ``xargs``, ``eval``, heredoc into ``sh`` ...). Each targets its own victim dir
  (relative path, so nothing rides on the "delete in root path" rule). Outcome read from disk plus
  the tool result the agent sent back to the provider. A twin run with
  ``approvals.single_query_mode: approve`` proves every spelling really deletes its victim, so a
  surviving victim means "gated", never "the variant was a no-op".
* Gateway: one real ``GatewayRunner`` child with a fake platform adapter (no approval buttons, so
  the production plain-text ``/approve`` / ``/deny`` flow). Several chats — separate sessions —
  hold a pending dangerous command at once; answering in one chat must settle only that chat's
  command, and ``/approve session`` in one chat must not pre-approve another.

``approvals.mode`` is pinned to ``manual`` (the shipped default ``smart`` asks an LLM guardian).
"""

from __future__ import annotations

import itertools
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict

import pytest

from tests.e2e.core.delivery._fake_platform import GatewayProcess, wait_until
from tests.e2e.core._pending_fixes import known_gate
from tests.e2e.core.security._helpers import BoundaryBreach, run_hermes, write_home
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell variants + process-group harness")

MANUAL = "approvals:\n  mode: manual\n  timeout: 120\n"

# name -> command template; ``{X}`` is the victim dir, relative to the terminal's cwd.
VARIANTS: Dict[str, str] = {
    "plain": "rm -rf {X}",
    "quoted": "'r''m' -rf {X}",
    "double_quoted": '"rm" -rf {X}',
    "backslash": "\\rm -rf {X}",
    "inner_backslash": "r\\m -rf {X}",
    "env": "env rm -rf {X}",
    "command_builtin": "command rm -rf {X}",
    "sh_c": 'sh -c "rm -rf {X}"',
    "bash_c": "bash -c 'rm -rf {X}'",
    "cmd_subst": "$(echo rm) -rf {X}",
    "backtick_subst": "`echo rm` -rf {X}",
    "param_default": "${{u:-rm}} -rf {X}",
    "ifs_split": "rm${{IFS}}-rf${{IFS}}{X}",
    "abs_path": "/bin/rm -rf {X}",
    "split_flags": "rm -r -f {X}",
    "long_flags": "rm --recursive --force {X}",
    "find_delete": "find {X} -delete",
    "find_exec": "find {X} -exec rm -rf {{}} +",
    "xargs": "echo {X} | xargs rm -rf",
    "eval": 'eval "rm -rf {X}"',
    "heredoc_sh": "sh <<'EOF'\nrm -rf {X}\nEOF",
    "pipe_sh": "printf 'rm -rf {X}' | sh",
    "chained": "true && rm -rf {X}",
    "subshell": "(rm -rf {X})",
    "python_c": "python3 -c \"import shutil; shutil.rmtree('{X}')\"",
}

# Detector misses confirmed live on origin/main: variant -> (pattern, "#issue symptom"), gated with
# ``known_gate(..., raises=BoundaryBreach)``. None today.
KNOWN: Dict[str, tuple[str, str]] = {}

CONTROL = "touch benign_marker"


# --- CLI ------------------------------------------------------------------------------------------


def _tool_results(requests: list[dict]) -> Dict[str, Dict[str, Any]]:
    """command -> parsed tool result the agent sent back to the provider (last request wins)."""
    commands: Dict[str, str] = {}
    results: Dict[str, Dict[str, Any]] = {}
    for body in requests:
        for msg in body.get("messages", []):
            for call in msg.get("tool_calls") or []:
                args = json.loads(call["function"]["arguments"] or "{}")
                commands[call["id"]] = args.get("command", "")
            if msg.get("role") == "tool" and msg.get("tool_call_id") in commands:
                results[commands[msg["tool_call_id"]]] = _parse_tool_content(msg["content"])
    return results


def _parse_tool_content(content: str) -> Dict[str, Any]:
    """The terminal tool's JSON payload; the loop may append plain-text notes after it."""
    text = content if isinstance(content, str) else json.dumps(content)
    try:
        payload, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except json.JSONDecodeError:
        payload = {}
    return {**(payload if isinstance(payload, dict) else {}), "_raw": text}


def _plant(root: Path) -> Path:
    victims = root / "victims"
    victims.mkdir(parents=True)
    for name in VARIANTS:
        (victims / name).mkdir()
        (victims / name / "keep.txt").write_text("sentinel", encoding="utf-8")
    return victims


def _single_query_run(root: Path, single_query_mode: str) -> Dict[str, Any]:
    """One ``hermes chat -q`` whose model calls ``terminal`` once per variant, then the control."""
    home, victims = root / "home", _plant(root)
    script: list[Any] = [ToolCall("terminal", {"command": t.format(X=n)}) for n, t in VARIANTS.items()]
    script += [ToolCall("terminal", {"command": CONTROL}), Text("done")]
    key = "sk-approval-cli"
    with FakeLLMServer(script, api_key=key) as srv:
        write_home(home / ".hermes", srv.base_url, api_key=key,
                   config=MANUAL + f"  single_query_mode: {single_query_mode}\n")
        proc = run_hermes(["chat", "-q", "run the maintenance commands", "-Q"], home, timeout=180, cwd=victims)
        requests = srv.main_requests()
    assert proc.returncode == 0, f"hermes chat -q failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    results = _tool_results(requests)
    missing = [n for n, t in VARIANTS.items() if t.format(X=n) not in results] + (
        [] if CONTROL in results else ["control"])
    assert not missing, f"the agent never reported a tool result for {missing}; stdout={proc.stdout[-1500:]}"
    return {"victims": victims, "results": results}


@pytest.fixture(scope="module")
def cli_deny(tmp_path_factory):
    return _single_query_run(tmp_path_factory.mktemp("approval-cli-deny"), "deny")


@pytest.fixture(scope="module")
def cli_approve(tmp_path_factory):
    return _single_query_run(tmp_path_factory.mktemp("approval-cli-approve"), "approve")


def test_cli_variants_are_live_deletes(cli_approve):
    """Discriminator for the table: approved, every spelling really deletes its victim."""
    victims = cli_approve["victims"]
    survived = [n for n in VARIANTS if (victims / n).exists()]
    assert not survived, f"variants that do not delete even when approved (fix the table): {survived}"
    control = cli_approve["results"][CONTROL]
    assert control.get("exit_code") == 0 and (victims / "benign_marker").exists(), control


def test_cli_benign_control_runs(cli_deny):
    """A harmless command runs in the same -q session that blocks the dangerous ones."""
    result = cli_deny["results"][CONTROL]
    assert result.get("exit_code") == 0, result
    assert (cli_deny["victims"] / "benign_marker").exists(), "benign control command did not run"


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_cli_dangerous_variant_blocked_without_approval(cli_deny, variant):
    victim = cli_deny["victims"] / variant
    result = cli_deny["results"][VARIANTS[variant].format(X=variant)]
    with known_gate(KNOWN, variant, raises=BoundaryBreach):
        if not (victim / "keep.txt").exists():
            raise BoundaryBreach(f"{variant!r} deleted its victim with nobody to approve it; tool result: {result}")
        if result.get("exit_code") == 0 or result.get("status") != "blocked":
            raise BoundaryBreach(f"{variant!r}: tool result not marked blocked: {result}")


# --- Gateway --------------------------------------------------------------------------------------

_CMD = re.compile(r"<<CMD>>(.*?)<<END>>", re.S)
PLATFORM = "fk_ap"


class ScriptedModel:
    """Turn a user message ``<<CMD>>...<<END>>`` into one terminal call; record each tool result."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.results: Dict[str, Dict[str, Any]] = {}

    def __call__(self, record: dict):
        msgs = record["body"]["messages"]
        if msgs and msgs[-1].get("role") == "tool":
            with self._lock:
                self.results.update(_tool_results([record["body"]]))
            return Text("finished")
        user = next((m for m in reversed(msgs) if m.get("role") == "user"), {})
        content = user.get("content")
        found = _CMD.search(content) if isinstance(content, str) else None
        return ToolCall("terminal", {"command": found.group(1)}) if found else Text("nothing to run")


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    model = ScriptedModel()
    root = tmp_path_factory.mktemp("approval-gw")
    with FakeLLMServer(model) as srv:
        gw = GatewayProcess(root, platforms={PLATFORM: "telegram"}, llm_base_url=srv.base_url,
                            extra_config=MANUAL + "updates:\n  check: false\n")
        gw.start()
        try:
            yield gw, model, root / "victims"
        finally:
            gw.stop()


_ids = itertools.count(1)


class Chat:
    def __init__(self, gw: GatewayProcess, model: ScriptedModel, victims: Path, chat_id: str) -> None:
        self.gw, self.model, self.chat_id = gw, model, chat_id
        self.victims = victims
        self.victims.mkdir(exist_ok=True)

    def say(self, text: str) -> None:
        self.gw.inject(PLATFORM, text, f"m{next(_ids)}", chat_id=self.chat_id)

    def victim(self, tag: str) -> Path:
        d = self.victims / f"{self.chat_id}-{tag}"
        d.mkdir()
        (d / "keep.txt").write_text("sentinel", encoding="utf-8")
        return d

    def run(self, command: str) -> str:
        self.say(f"please run this <<CMD>>{command}<<END>>")
        return command

    def texts(self) -> list[str]:
        return [m.text for m in self.gw.platform_view().visible(PLATFORM, self.chat_id)]

    def prompts(self) -> int:
        return sum("/approve" in t and "/deny" in t for t in self.texts())

    def await_pending(self, victim: Path, command: str) -> None:
        """Wait for this chat's approval prompt (or an early settle, which ``_pending`` reports)."""
        wait_until(lambda: self.prompts() >= 1 or command in self.model.results,
                   f"approval prompt for {command!r} in {self.chat_id}", timeout=60,
                   proc=self.gw.proc, log=self.gw.log)
        _pending(self, victim, command)

    def wait_result(self, command: str) -> Dict[str, Any]:
        wait_until(lambda: command in self.model.results, f"tool result for {command!r}", timeout=60,
                   proc=self.gw.proc, log=self.gw.log)
        self.gw.wait_idle([self.chat_id], f"{self.chat_id} idle")
        return self.model.results[command]

    def blocked_on_approval(self) -> bool:
        return not self.gw.idle([self.chat_id])


def _chats(gateway, *names: str) -> list[Chat]:
    gw, model, victims = gateway
    return [Chat(gw, model, victims, name) for name in names]


def _pending(chat: Chat, victim: Path, command: str) -> None:
    """``chat`` holds ``command`` pending: prompt shown, victim intact, turn still blocked."""
    if not (victim / "keep.txt").exists():
        raise BoundaryBreach(f"{chat.chat_id}: {command!r} ran without an approval in its own chat")
    if command in chat.model.results:
        raise BoundaryBreach(f"{chat.chat_id}: {command!r} was settled without an answer in its own chat: "
                             f"{chat.model.results[command]}")
    assert chat.blocked_on_approval(), f"{chat.chat_id} is no longer waiting on its approval: {chat.texts()}"


def _denied(victim: Path, result: Dict[str, Any]) -> None:
    if not (victim / "keep.txt").exists() or result.get("exit_code") == 0:
        raise BoundaryBreach(f"denied command ran: {result}")
    assert result.get("status") == "blocked", result


def test_gateway_approve_once_settles_only_its_own_chat(gateway):
    a, b, c = _chats(gateway, "once-a", "once-b", "once-c")
    va, vb = a.victim("x"), b.victim("x")
    cmd_a, cmd_b = a.run(f"rm -rf {va}"), b.run(f"bash -c 'rm -rf {vb}'")
    a.await_pending(va, cmd_a)
    b.await_pending(vb, cmd_b)

    c.say("/approve")  # a chat with nothing pending must not answer anyone else's prompt
    wait_until(lambda: c.texts(), "reply to /approve in the idle chat", timeout=30)
    _pending(a, va, cmd_a)
    _pending(b, vb, cmd_b)

    a.say("/approve")
    result_a = a.wait_result(cmd_a)
    assert not va.exists() and result_a.get("exit_code") == 0, f"approved command did not run: {result_a}"
    _pending(b, vb, cmd_b)

    b.say("/deny")
    _denied(vb, b.wait_result(cmd_b))


def test_gateway_deny_settles_only_its_own_chat(gateway):
    a, b = _chats(gateway, "deny-a", "deny-b")
    va, vb = a.victim("x"), b.victim("x")
    cmd_a, cmd_b = a.run(f'sh -c "rm -rf {va}"'), b.run(f"rm -rf {vb}")
    a.await_pending(va, cmd_a)
    b.await_pending(vb, cmd_b)

    a.say("/deny not now")
    _denied(va, a.wait_result(cmd_a))
    _pending(b, vb, cmd_b)

    b.say("/approve")
    result_b = b.wait_result(cmd_b)
    assert not vb.exists() and result_b.get("exit_code") == 0, f"approved command did not run: {result_b}"


def test_gateway_session_scope_does_not_leak_to_other_chat(gateway):
    a, b = _chats(gateway, "scope-a", "scope-b")
    va1, va2, vb = a.victim("1"), a.victim("2"), b.victim("1")

    cmd_a1 = a.run(f"rm -rf {va1}")
    a.await_pending(va1, cmd_a1)
    a.say("/approve session")
    assert a.wait_result(cmd_a1).get("exit_code") == 0 and not va1.exists()

    # Control: the session grant is real — the same pattern in the SAME chat runs unprompted.
    cmd_a2 = a.run(f"rm -rf {va2}")
    wait_until(lambda: a.prompts() >= 2 or cmd_a2 in a.model.results, f"{cmd_a2!r} to run or prompt again",
               timeout=60, proc=a.gw.proc, log=a.gw.log)
    assert a.prompts() == 1, f"/approve session did not stick in its own chat: {a.texts()}"
    result_a2 = a.wait_result(cmd_a2)
    assert result_a2.get("exit_code") == 0 and not va2.exists(), result_a2

    cmd_b = b.run(f"rm -rf {vb}")
    b.await_pending(vb, cmd_b)  # chat A's session grant must not pre-approve chat B
    b.say("/deny")
    _denied(vb, b.wait_result(cmd_b))
