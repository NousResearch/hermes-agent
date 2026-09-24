"""Private harness for test_secret_redaction*.py: scenario table, scripted model, sink collection.

Secrets are allocated so that every scenario owns disjoint values: the values a scenario leaves raw
BY DESIGN (a tool argument, the user's own prompt) never coincide with the values another scenario
asserts are masked, so one Hermes home can serve every scenario and each sink is scanned once.
"""

from __future__ import annotations

import json
import re
import secrets
import string
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tests.e2e.core.delivery._fake_platform import GatewayProcess, read_jsonl, wait_until
from tests.e2e.core.security._helpers import REPO_ROOT, db_blob, files_containing, run_hermes
from tests.fakes.fake_llm_provider import Error, Response, Text, ToolCall


def _alnum(n: int) -> str:
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(n))


@dataclass(frozen=True)
class Secrets:
    """Every secret value one Hermes home sees; see the module docstring for the allocation."""

    provider: str = field(default_factory=lambda: "sk-proj-" + _alnum(40))   # .env OPENAI_API_KEY
    env_opaque: str = field(default_factory=lambda: secrets.token_hex(20))   # .env ACME_SERVICE_TOKEN
    env_github: str = field(default_factory=lambda: "ghp_" + _alnum(36))     # .env GITHUB_TOKEN
    echo_github: str = field(default_factory=lambda: "ghp_" + _alnum(36))    # only in a workspace file
    echo_slack: str = field(default_factory=lambda: f"xoxb-{secrets.randbelow(10**12):012d}-{_alnum(24)}")
    echo_openai: str = field(default_factory=lambda: "sk-" + _alnum(40))
    echo_bearer: str = field(default_factory=lambda: _alnum(40))             # opaque, no vendor prefix
    user_key: str = field(default_factory=lambda: "sk-" + _alnum(44))        # pasted by the user
    nonce: str = field(default_factory=lambda: "nonce" + secrets.token_hex(6))

    def env(self) -> dict[str, str]:
        return {"ACME_SERVICE_TOKEN": self.env_opaque, "GITHUB_TOKEN": self.env_github}


# The user explicitly runs without approval prompts, so the credential-in-command scan does not block the
# tool-argument scenario before the command executes (that guard is covered by the approvals lane).
CONFIG = "approvals:\n  mode: \"off\"\n"


def core(value: str) -> str:
    """The part of a secret no mask may keep: a head/tail mask (``sk-pro...AbCd``) is not a leak."""
    return value[6:-4]


# Sinks a scenario is held to; the table below documents which ones the contract covers.
LOGS, STORE, EXPORT, EXPORT_REDACTED, WIRE, PLATFORM = (
    "logs", "store", "export", "export --redact", "next provider request", "platform wire")
CONTENT_SINKS = (LOGS, STORE, EXPORT, EXPORT_REDACTED, WIRE, PLATFORM)
# The session database keeps tool arguments and the user's own prompt as executed (docs:
# user-guide/security.md "The session database itself still holds the command as it was executed");
# the provider replay of the model's own tool call and the default (non --redact) export read it back.
RAW_BY_DESIGN_SINKS = (LOGS, EXPORT_REDACTED, PLATFORM)


@dataclass(frozen=True)
class Scenario:
    name: str
    secrets: Callable[[Secrets], list[str]]
    script: Callable[["Ctx"], list[Response]]
    sinks: tuple[str, ...]
    prompt: str = "please run the task"
    followup: bool = False  # a second user turn, so the NEXT request after the answer exists


@dataclass
class Ctx:
    """What a scenario script may reference: the secrets, the workspace, the provider port, the .env."""

    keys: Secrets
    ws: Path
    env_file: Path
    port: int = 0


def _tee(c: Ctx, name: str, src: "str | Path") -> str:
    return f"cat {src} | tee {c.ws / name}"


SCENARIOS: dict[str, Scenario] = {s.name: s for s in (
    # (a) the terminal tool prints secret-shaped strings that are NOT in the env
    Scenario("tool_echoes_prefixed", lambda k: [k.echo_github, k.echo_slack, k.echo_openai, k.echo_bearer],
             lambda c: [ToolCall("terminal", {"command": _tee(c, "echoed_a.txt", c.ws / "fixture.txt")}),
                                  Text("printed the fixture")], CONTENT_SINKS),
    # (b) the tool prints the profile's real .env (provider key + an opaque token) and a plain file with the key
    Scenario("tool_reads_env_file", lambda k: [k.provider, k.env_opaque],
             lambda c: [ToolCall("terminal", {"command": _tee(c, "echoed_b.txt", c.env_file)}),
                                  ToolCall("terminal", {"command": _tee(c, "echoed_b2.txt", c.ws / "notes.txt")}),
                                  Text("read both files")], CONTENT_SINKS),
    # (c) the model's own answer contains the provider key
    Scenario("assistant_text", lambda k: [k.provider],
             lambda c: [Text(f"Your key is {c.keys.provider} - keep it safe.")], CONTENT_SINKS,
             followup=True),
    # (d) the key is a tool ARGUMENT (curl -H 'Authorization: Bearer KEY') to a local endpoint
    Scenario("tool_arg", lambda k: [k.env_github],
             lambda c: [ToolCall("terminal", {"command": (
                 f"curl -s -o {c.ws / 'models.json'} -H 'Authorization: Bearer {c.keys.env_github}' "
                 f"http://127.0.0.1:{c.port}/v1/models")}), Text("called the endpoint")], RAW_BY_DESIGN_SINKS),
    # (e) the user pastes a key into the prompt (the turn preview is logged)
    Scenario("user_prompt", lambda k: [k.user_key], lambda c: [Text("noted")], RAW_BY_DESIGN_SINKS),
    # (f) the provider echoes the request's key back in an error body
    Scenario("provider_error_echo", lambda k: [k.provider],
             lambda c: [Error(401, f"Incorrect API key provided: {c.keys.provider} ({c.keys.nonce})")],
             (LOGS, STORE, EXPORT, EXPORT_REDACTED, PLATFORM)),
)}


def prompt_for(name: str, k: Secrets, *, followup: bool = False) -> str:
    tag = f"[scn:{name}{':followup' if followup else ''}]"
    if name == "user_prompt":
        return f"{tag} {k.nonce} my key is {k.user_key}"
    return f"{tag} {'thanks, anything else?' if followup else SCENARIOS[name].prompt}"


_TAG_RE = re.compile(r"\[scn:([a-z_]+)(:followup)?\]")


def _text(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


class Director:
    """Scripted model: the newest user message's ``[scn:<name>]`` tag selects the script and the
    number of assistant messages since it selects the step."""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx

    def __call__(self, record: dict) -> Response:
        msgs = record["body"].get("messages", [])
        last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
        m = _TAG_RE.search(_text(msgs[last_user])) if last_user >= 0 else None
        if m is None:
            return Text("no scenario")
        if m.group(2):
            return Text(f"nothing else for {m.group(1)}")
        step = sum(1 for x in msgs[last_user + 1:] if x.get("role") == "assistant")
        script = SCENARIOS[m.group(1)].script(self.ctx)
        return script[min(step, len(script) - 1)] if isinstance(script[-1], Error) or step < len(script) \
            else Text(f"done {m.group(1)}")


def seed_workspace(ws: Path, k: Secrets) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "fixture.txt").write_text(
        f"github {k.echo_github}\nslack {k.echo_slack}\nopenai {k.echo_openai}\n"
        f"Authorization: Bearer {k.echo_bearer}\n", encoding="utf-8")
    (ws / "notes.txt").write_text(f"the deploy key is {k.provider}\n", encoding="utf-8")


def echo_preconditions(ws: Path, k: Secrets, llm_gets: list[dict], logs: str) -> dict[str, list[str]]:
    """Per scenario: the evidence that the secret really reached the tool/log path (harness facts)."""
    def has(name: str, *values: str) -> list[str]:
        path = ws / name
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        return [f"{name} lacks {v[:8]}…" for v in values if v not in text]

    auths = {g.get("auth") for g in llm_gets}
    return {
        "tool_echoes_prefixed": has("echoed_a.txt", k.echo_github, k.echo_slack, k.echo_openai, k.echo_bearer),
        "tool_reads_env_file": has("echoed_b.txt", k.provider, k.env_opaque) + has("echoed_b2.txt", k.provider),
        "assistant_text": [],
        "tool_arg": [] if f"Bearer {k.env_github}" in auths else ["curl never reached the endpoint with the key"],
        "user_prompt": [] if k.nonce in logs else ["the turn preview never reached the logs"],
        "provider_error_echo": [] if k.nonce in logs else ["the provider error never reached the logs"],
    }


@dataclass
class Sinks:
    """Raw text of every persisted/egress sink of one Hermes home, keyed by sink name."""

    texts: dict[str, dict[str, str]] = field(default_factory=dict)  # sink -> {location: text}

    def hits(self, sink: str, needles: list[str], only: str = "") -> list[str]:
        """``only`` narrows a per-chat sink (the platform journal) to one scenario's chat."""
        out = []
        for loc, text in self.texts.get(sink, {}).items():
            if only and loc != only and sink == PLATFORM:
                continue
            for n in needles:
                at = text.find(core(n))
                if at >= 0:
                    ctx = text[max(0, at - 80):at + len(n) + 20].replace(core(n), "<SECRET>")
                    out.append(f"{sink} [{loc}] holds {n[:10]}…: …{ctx!r}…")
        return out


def collect(home: Path, requests: list[dict], platform_journal: Path | None = None) -> Sinks:
    hh = home / ".hermes"
    logs = {str(p.relative_to(hh / "logs")): p.read_text(encoding="utf-8", errors="replace")
            for p in sorted((hh / "logs").rglob("*")) if p.is_file()}
    store = {"state.db (decoded rows)": db_blob(hh / "state.db")}
    for p in sorted(hh.rglob("*")):  # raw bytes: WAL, FTS shadow pages, sessions/, caches
        if p.is_file() and p.name != ".env" and "logs" not in p.relative_to(hh).parts:
            store[str(p.relative_to(hh))] = p.read_bytes().decode("utf-8", "replace")
    out = home / "exports"
    out.mkdir(exist_ok=True)
    exports = {}
    for flag, name in (([], "plain.jsonl"), (["--redact"], "redacted.jsonl")):
        r = run_hermes(["sessions", "export", str(out / name), *flag], home, timeout=90)
        assert r.returncode == 0 and (out / name).exists(), f"sessions export failed: {r.stdout}\n{r.stderr}"
        exports[name] = (out / name).read_text(encoding="utf-8")
    wire = {f"request #{i} ({r['kind']})": json.dumps(r["body"]) for i, r in enumerate(requests)
            if r["kind"] in ("main", "aux")}
    platform: dict[str, str] = {}  # every post AND every intermediate edit (streamed text), per chat
    chat_of: dict[str, str] = {}
    for rec in read_jsonl(platform_journal) if platform_journal is not None else []:
        chat = chat_of.setdefault(rec["message_id"], str(rec.get("chat_id", "?")))
        platform[f"chat {chat}"] = platform.get(f"chat {chat}", "") + json.dumps(rec) + "\n"
    return Sinks({LOGS: logs, STORE: store, EXPORT: {"plain.jsonl": exports["plain.jsonl"]},
                  EXPORT_REDACTED: {"redacted.jsonl": exports["redacted.jsonl"]}, WIRE: wire, PLATFORM: platform})


def leak_report(sinks: Sinks, scenario: Scenario, keys: Secrets) -> list[str]:
    needles = scenario.secrets(keys)
    return [hit for sink in scenario.sinks for hit in sinks.hits(sink, needles, only=f"chat {chat_for(scenario.name)}")]


def chat_for(name: str) -> str:
    return f"c-{name}"


def assert_harness_sane(sinks: Sinks, *, gateway: bool = False) -> None:
    """A sink that is empty cannot prove absence: every scanned sink must actually exist."""
    for sink in (LOGS, STORE, EXPORT, EXPORT_REDACTED, WIRE) + ((PLATFORM,) if gateway else ()):
        assert any(t.strip() for t in sinks.texts.get(sink, {}).values()), f"sink {sink!r} is empty"
    wanted = {"agent.log", "errors.log"} | ({"gateway.log"} if gateway else set())
    assert wanted <= set(sinks.texts[LOGS]), sorted(sinks.texts[LOGS])


@dataclass
class World:
    """One Hermes home after every scenario ran: its sinks and each scenario's travel evidence."""

    keys: Secrets
    sinks: Sinks
    pre: dict[str, list[str]]
    runs: dict[str, str]


def check(world: World, scenario: str) -> None:
    """Harness precondition (plain AssertionError), then the boundary (BoundaryBreach, KNOWN-able)."""
    from tests.e2e.core.security._helpers import BoundaryBreach

    missing = world.pre[scenario]
    assert not missing, (f"{scenario}: the secret never travelled, so absence proves nothing: {missing}\n"
                         f"{world.runs[scenario]}")
    leaks = leak_report(world.sinks, SCENARIOS[scenario], world.keys)
    if leaks:
        raise BoundaryBreach(f"{scenario}: secret reached a redacted sink:\n  " + "\n  ".join(leaks))


# Gateway ------------------------------------------------------------------------------------------


class LoggingGateway(GatewayProcess):
    """The delivery suite's real GatewayRunner child, with the gateway's file logging installed the way
    ``start_gateway`` does (agent.log / errors.log / gateway.log under the child's HERMES_HOME)."""

    def start(self) -> "LoggingGateway":
        assert self.proc is None
        self.boots += 1
        ready_before = len(read_jsonl(self.spool / "ready.jsonl"))
        with open(self.log, "wb") as log:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "tests.e2e.core.security._redact", "serve", str(self.spool)],
                cwd=str(REPO_ROOT), env=self.env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.pids.append(self.proc.pid)
        pid = self.proc.pid
        rec = wait_until(lambda: next((r for r in read_jsonl(self.spool / "ready.jsonl")[ready_before:]
                                       if r["pid"] == pid), None),
                         "gateway child ready", timeout=240, proc=self.proc, log=self.log)
        import socket
        self.sock = socket.create_connection(("127.0.0.1", rec["port"]), timeout=120)
        self._rfile = self.sock.makefile("rb")
        return self


def files_with(root: Path, needles: list[str]) -> list[str]:
    return files_containing(root, [core(n) for n in needles])


if __name__ == "__main__":  # pragma: no cover - gateway child entry
    if len(sys.argv) >= 3 and sys.argv[1] == "serve":
        from hermes_logging import setup_logging

        setup_logging(mode="gateway")
        from tests.e2e.core.delivery._fake_platform import _serve

        _serve(Path(sys.argv[2]))
