"""Plugin slash commands typed in the classic CLI run in the gateway (#134889 PLG-7, CLI-13).

The plugin's hooks, tools and MCP servers live in the gateway, so its slash command runs there too
(``slash.exec``) and ``ctx.call_mcp`` reaches the connected servers. The CLI learns the command names
from the gateway's ``commands.catalog``: no plugin code is imported or executed in the CLI process.
"""
import pytest

import cli as cli_mod
from cli import HermesCLI
from hermes_cli.commands_completion import SlashCommandCompleter
from hermes_cli.gateway_client import GatewayClientError


class _Conn:
    closed = False
    final = False
    epoch = 1

    def __init__(self, plugins, reply=None, catalog_error=None):
        self.calls = []
        self._plugins = plugins
        self._reply = reply or {"type": "exec", "output": "mcp says hi"}
        self.catalog_error = catalog_error
        self.extra = {}  # merged into the catalog reply (e.g. plugin_discovery_error)
        self.after_catalog = None  # runs once the reply exists, before the caller sees it

    def call(self, method, **params):
        return self.call_tagged(method, **params)[0]

    def call_until(self, method, _wake, **params):  # slash.exec: no answer deadline
        return self.call_tagged(method, **params)[0]

    def call_tagged(self, method, **params):
        self.calls.append((method, params))
        if self.closed:  # a hung-up connection redials on its next call (gateway_connection)
            self.closed, self.epoch = False, self.epoch + 1
        epoch = self.epoch  # the socket that carried this request
        if method == "commands.catalog":
            if self.catalog_error is not None:
                raise self.catalog_error
            rows = [["/" + name, desc] for name, desc in self._plugins.items()]
            reply = {"pairs": [["/help", "Show help"], *rows],
                     "categories": [{"name": "Session", "pairs": [["/help", "Show help"]]},
                                    {"name": "Plugin commands", "pairs": rows}], **self.extra}
            hook, self.after_catalog = self.after_catalog, None
            if hook:
                hook()
            return reply, epoch
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply, epoch


@pytest.fixture
def local_plugin_reads(monkeypatch):
    """Every way the CLI process could load or run plugin code; each records the call."""
    reads = []

    def recorder(name):
        def read(*_a, **_k):
            reads.append(name)
            return {} if name == "get_plugin_commands" else None
        return read
    for name in ("get_plugin_commands", "get_plugin_command_handler", "discover_plugins"):
        monkeypatch.setattr(f"hermes_cli.plugins.{name}", recorder(name))
    return reads


def _cli(monkeypatch, conn):
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: conn)
    monkeypatch.setattr(cli_mod, "_ensure_skill_commands", dict)
    monkeypatch.setattr(cli_mod, "get_skill_bundles", dict)
    c = HermesCLI.__new__(HermesCLI)
    c.config = {}
    c.session_id = "local-s"
    return c


def test_plugin_command_runs_in_the_gateway(monkeypatch, capsys, local_plugin_reads):
    conn = _Conn({"mcp-status": "Ask the MCP server"})
    c = _cli(monkeypatch, conn)
    assert c._process_unregistered_slash("/mcp-status Deep Check", "/mcp-status deep check") is True
    assert ("slash.exec", {"session_id": "local-s", "command": "mcp-status Deep Check"}) in conn.calls
    assert "mcp says hi" in capsys.readouterr().out
    assert local_plugin_reads == []


def test_gateway_refusal_is_printed_not_run_locally(monkeypatch, capsys, local_plugin_reads):
    conn = _Conn({"mcp-status": "Ask the MCP server"}, reply=GatewayClientError("permission_denied"))
    c = _cli(monkeypatch, conn)
    assert c._process_unregistered_slash("/mcp-status", "/mcp-status") is True
    assert "permission_denied" in capsys.readouterr().out
    assert local_plugin_reads == []


def test_non_plugin_command_falls_through_without_loading_plugins(monkeypatch, local_plugin_reads):
    conn = _Conn({"mcp-status": "Ask the MCP server"})
    c = _cli(monkeypatch, conn)
    expanded = []
    monkeypatch.setattr(HermesCLI, "_expand_slash_prefix", lambda self, *a: expanded.append(a) or True)
    assert c._process_unregistered_slash("/zzz", "/zzz") is True
    assert expanded and not any(method == "slash.exec" for method, _ in conn.calls)
    assert local_plugin_reads == []


def _catalog_reads(conn):
    return [m for m, _ in conn.calls].count("commands.catalog")


def _execs(conn):
    return [p["command"] for m, p in conn.calls if m == "slash.exec"]


def test_known_plugin_commands_reuse_the_sessions_catalog(monkeypatch, local_plugin_reads):
    conn = _Conn({"mcp-status": "Ask the MCP server"})
    c = _cli(monkeypatch, conn)
    for _ in range(3):
        c._process_unregistered_slash("/mcp-status", "/mcp-status")
    assert _catalog_reads(conn) == 1 and _execs(conn) == ["mcp-status"] * 3


def test_a_plugin_the_gateway_gained_dispatches_after_one_miss_refresh(monkeypatch, local_plugin_reads):
    """The same open CLI: a plugin enabled in the gateway after the first read is not 'unknown'."""
    conn = _Conn({})
    c = _cli(monkeypatch, conn)
    expanded = []
    monkeypatch.setattr(HermesCLI, "_expand_slash_prefix", lambda self, *a: expanded.append(a) or True)
    c._process_unregistered_slash("/zzz", "/zzz")
    reads = _catalog_reads(conn)
    conn._plugins["late-plugin"] = "Enabled after the CLI opened"
    assert c._process_unregistered_slash("/late-plugin now", "/late-plugin now") is True
    assert _execs(conn) == ["late-plugin now"]
    assert _catalog_reads(conn) == reads + 1  # one refresh on the miss, no loop
    assert len(expanded) == 1 and local_plugin_reads == []


def test_an_unknown_command_refreshes_the_catalog_once_then_falls_through(monkeypatch):
    conn = _Conn({"mcp-status": "Ask the MCP server"})
    c = _cli(monkeypatch, conn)
    expanded = []
    monkeypatch.setattr(HermesCLI, "_expand_slash_prefix", lambda self, *a: expanded.append(a) or True)
    c._process_unregistered_slash("/mcp-status", "/mcp-status")
    for submitted in range(1, 4):  # each repeat of the unknown name: exactly one read, no retry loop
        c._process_unregistered_slash("/zzz", "/zzz")
        assert _catalog_reads(conn) == 1 + submitted and len(expanded) == submitted
    assert _execs(conn) == ["mcp-status"]


@pytest.mark.parametrize("boundary", ["new_session", "reconnect"])
def test_a_session_boundary_rereads_the_catalog(monkeypatch, boundary):
    """/new or resume (a new session id) and a redial to another gateway life re-read the names,
    so completion also sees a plugin set the gateway changed meanwhile."""
    from hermes_cli import gateway_plugin_commands
    conn = _Conn({"old-plugin": "Before"})
    c = _cli(monkeypatch, conn)
    assert set(gateway_plugin_commands.names(c) or ()) == {"old-plugin"}
    conn._plugins = {"new-plugin": "After a gateway restart"}
    assert set(gateway_plugin_commands.names(c) or ()) == {"old-plugin"}  # completion keeps the session's read
    if boundary == "new_session":
        c.session_id = "local-s2"
    else:
        conn.epoch += 1
    assert set(gateway_plugin_commands.names(c) or ()) == {"new-plugin"}
    c._process_unregistered_slash("/new-plugin", "/new-plugin")
    assert _execs(conn) == ["new-plugin"]


def test_a_failed_catalog_read_runs_nothing_else(monkeypatch, capsys, local_plugin_reads):
    """Not knowing the gateway's plugin commands is not evidence a name is no plugin: a same-named
    skill or bundle must not run in its place, nor prefix expansion or a local plugin."""
    conn = _Conn({}, catalog_error=GatewayClientError("gateway restarting"))
    c = _cli(monkeypatch, conn)
    ran = []
    monkeypatch.setattr(cli_mod, "_ensure_skill_commands", lambda: {"/mcp-status": {"name": "mcp-status"}})
    monkeypatch.setattr(cli_mod, "get_skill_bundles", lambda: {"/mcp-status": {"skills": []}})
    for name in ("_run_skill_bundle_command", "_run_skill_slash_command", "_expand_slash_prefix"):
        monkeypatch.setattr(HermesCLI, name, lambda self, *a, _n=name: ran.append(_n) or True)
    assert c._process_unregistered_slash("/mcp-status", "/mcp-status") is True
    assert "gateway restarting" in capsys.readouterr().out
    assert ran == [] and _execs(conn) == [] and local_plugin_reads == []


def _colliding_skill_and_bundle(monkeypatch, name):
    ran = []
    monkeypatch.setattr(cli_mod, "_ensure_skill_commands", lambda: {f"/{name}": {"name": name}})
    monkeypatch.setattr(cli_mod, "get_skill_bundles", lambda: {f"/{name}": {"skills": []}})
    for meth in ("_run_skill_bundle_command", "_run_skill_slash_command", "_expand_slash_prefix"):
        monkeypatch.setattr(HermesCLI, meth, lambda self, *a, _n=meth: ran.append(_n) or True)
    return ran


def test_a_catalog_whose_plugin_discovery_failed_runs_nothing_else(monkeypatch, capsys, local_plugin_reads):
    """The gateway answers commands.catalog but says its plugin discovery failed: the missing plugin
    list is not evidence the name is no plugin, so the colliding skill or bundle must not run."""
    from hermes_cli import gateway_plugin_commands
    conn = _Conn({})
    conn.extra = {"plugin_discovery_error": "boom in register()",
                  "warning": "plugin command discovery unavailable: boom in register()"}
    c = _cli(monkeypatch, conn)
    ran = _colliding_skill_and_bundle(monkeypatch, "mcp-status")
    assert c._process_unregistered_slash("/mcp-status", "/mcp-status") is True
    assert "boom in register()" in capsys.readouterr().out
    assert ran == [] and _execs(conn) == [] and local_plugin_reads == []
    assert gateway_plugin_commands.names(c) == {}  # completion degrades quietly
    assert capsys.readouterr().out == ""
    conn.extra = {}  # discovery recovered: the degraded answer was never cached
    conn._plugins = {"mcp-status": "Ask the MCP server"}
    assert c._process_unregistered_slash("/mcp-status", "/mcp-status") is True
    assert _execs(conn) == ["mcp-status"] and ran == []


def test_an_older_gateway_without_the_discovery_field_keeps_todays_behaviour(monkeypatch):
    conn = _Conn({})
    conn.extra = {"warning": "skill discovery unavailable: x"}  # no plugin_discovery_error key
    c = _cli(monkeypatch, conn)
    ran = _colliding_skill_and_bundle(monkeypatch, "probe")
    assert c._process_unregistered_slash("/probe", "/probe") is True
    assert ran == ["_run_skill_bundle_command"]


def test_a_plugin_command_a_display_entry_hides_still_dispatches(monkeypatch, local_plugin_reads):
    """Ink's ``/logs`` row hides a plugin's ``/logs`` from the display categories; the catalog's
    ``plugin_commands`` still names it, and that list routes and completes."""
    from hermes_cli import gateway_plugin_commands
    conn = _Conn({})
    conn.extra = {"plugin_commands": {"logs": "Plugin logs"}}
    c = _cli(monkeypatch, conn)
    ran = _colliding_skill_and_bundle(monkeypatch, "logs")
    assert c._process_unregistered_slash("/logs", "/logs") is True
    assert _execs(conn) == ["logs"] and ran == [] and local_plugin_reads == []
    assert gateway_plugin_commands.names(c) == {"logs": "Plugin logs"}


def test_a_hung_up_connection_rereads_the_catalog_before_a_cached_name_runs(monkeypatch, local_plugin_reads):
    """A hung-up connection keeps its epoch until its next call redials, so the cache still matched;
    the gateway that answers next may have lost the plugin. Its catalog decides, not the stale one."""
    conn = _Conn({"gone-plugin": "Before the restart"})
    c = _cli(monkeypatch, conn)
    c._process_unregistered_slash("/gone-plugin", "/gone-plugin")
    conn.closed, conn._plugins = True, {}  # the gateway restarted without the plugin
    ran = _colliding_skill_and_bundle(monkeypatch, "gone-plugin")
    assert c._process_unregistered_slash("/gone-plugin", "/gone-plugin") is True
    # The name's miss re-reads the catalog (redialing) and the session's skills come from that same
    # read: one refreshed snapshot, no stale cache answers.
    assert _execs(conn) == ["gone-plugin"] and _catalog_reads(conn) == 2
    assert ran == ["_run_skill_bundle_command"] and local_plugin_reads == []


def test_a_hung_up_connection_rereads_the_sessions_skills(monkeypatch):
    """/help and completion: the old gateway life's skills are not served after it hung up."""
    from hermes_cli import gateway_plugin_commands
    conn = _Conn({})
    conn.extra = {"skills": ["/old-skill"]}
    c = _cli(monkeypatch, conn)
    assert set(gateway_plugin_commands.skills(c) or ()) == {"/old-skill"}
    conn.closed, conn.extra = True, {"skills": ["/new-skill"]}  # same epoch until the next call redials
    assert set(gateway_plugin_commands.skills(c) or ()) == {"/new-skill"}
    assert _catalog_reads(conn) == 2


def test_a_hung_up_connection_rereads_the_plugin_toolsets(monkeypatch):
    """/tools validation: the old gateway life's plugin toolsets are not served after it hung up."""
    from hermes_cli import gateway_plugin_commands
    conn = _Conn({}, reply={"toolsets": [{"key": "old-ts"}]})
    c = _cli(monkeypatch, conn)
    assert gateway_plugin_commands.toolsets(c) == ([("old-ts", "old-ts", "")], set())
    conn.closed, conn._reply = True, {"toolsets": [{"key": "new-ts"}]}
    assert gateway_plugin_commands.toolsets(c) == ([("new-ts", "new-ts", "")], set())
    assert [m for m, _ in conn.calls].count("plugins.list") == 2


def test_a_reply_that_is_no_plugin_run_is_reported(monkeypatch, capsys):
    """The gateway lost the plugin between lookup and ``slash.exec`` and answered for a same-named
    skill (``type: skill``): nothing ran, and the input must not vanish without a word."""
    conn = _Conn({"gone-plugin": "Before the restart"}, reply={"type": "skill", "message": "skill body"})
    c = _cli(monkeypatch, conn)
    assert c._process_unregistered_slash("/gone-plugin", "/gone-plugin") is True
    out = capsys.readouterr().out
    assert "/gone-plugin" in out and "nothing was run" in out and "skill body" not in out


def test_a_catalog_served_before_a_redial_is_not_cached_as_the_new_lifes(monkeypatch):
    """Completion worker: the reply comes from epoch E, another caller redials to E+1 before the worker
    publishes it. The cache must not take E's names as E+1's; the next lookup reads again."""
    from hermes_cli import gateway_plugin_commands
    conn = _Conn({"old-plugin": "Served by the old gateway life"})
    c = _cli(monkeypatch, conn)

    def redial():
        conn.epoch += 1
        conn._plugins = {"new-plugin": "Served by the new gateway life"}
    conn.after_catalog = redial
    assert set(gateway_plugin_commands.names(c) or ()) == {"old-plugin"}  # this caller's own answer
    assert set(gateway_plugin_commands.names(c) or ()) == {"new-plugin"}
    assert _catalog_reads(conn) == 2
    c._process_unregistered_slash("/new-plugin", "/new-plugin")
    assert _execs(conn) == ["new-plugin"] and _catalog_reads(conn) == 2


def test_a_catalog_read_across_a_session_change_is_not_cached_for_the_new_session(monkeypatch):
    from hermes_cli import gateway_plugin_commands
    conn = _Conn({"old-plugin": "Before"})
    c = _cli(monkeypatch, conn)

    def switch():
        c.session_id = "local-s2"
        conn._plugins = {"new-plugin": "After"}
    conn.after_catalog = switch
    gateway_plugin_commands.names(c)
    assert set(gateway_plugin_commands.names(c) or ()) == {"new-plugin"}
    assert _catalog_reads(conn) == 2


def test_slash_new_rereads_the_catalog(monkeypatch):
    """The real /new path (``new_session``), not a hand-set session id. Off a GatewaySessionStore the
    id is main's local mint; the gateway's session.create path is covered by its own tests."""
    from hermes_cli import gateway_plugin_commands
    conn = _Conn({"old-plugin": "Before"})
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: conn)
    c = HermesCLI()
    c.agent = None
    assert set(gateway_plugin_commands.names(c) or ()) == {"old-plugin"}
    conn._plugins = {"new-plugin": "After"}
    before = c.session_id
    c.new_session(silent=True)
    assert c.session_id != before
    assert set(gateway_plugin_commands.names(c) or ()) == {"new-plugin"}
    assert _catalog_reads(conn) == 2


def test_a_failed_refresh_on_a_miss_runs_nothing_else(monkeypatch, capsys):
    conn = _Conn({"mcp-status": "Ask the MCP server"})
    c = _cli(monkeypatch, conn)
    c._process_unregistered_slash("/mcp-status", "/mcp-status")
    conn.catalog_error = GatewayClientError("gateway restarting")
    ran = []
    monkeypatch.setattr(cli_mod, "_ensure_skill_commands", lambda: {"/probe-skill": {"name": "probe-skill"}})
    monkeypatch.setattr(HermesCLI, "_run_skill_slash_command", lambda self, *a: ran.append(a))
    assert c._process_unregistered_slash("/probe-skill", "/probe-skill") is True
    assert "gateway restarting" in capsys.readouterr().out and ran == []


def test_completion_degrades_quietly_when_the_catalog_fails(monkeypatch, capsys):
    from hermes_cli import gateway_plugin_commands
    c = _cli(monkeypatch, _Conn({}, catalog_error=GatewayClientError("gateway restarting")))
    assert gateway_plugin_commands.names(c) == {}
    assert capsys.readouterr().out == ""


def test_completion_lists_the_gateways_plugin_commands(local_plugin_reads):
    from prompt_toolkit.document import Document
    completer = SlashCommandCompleter(plugin_commands_provider=lambda: {"mcp-status": "Ask the MCP server"})
    found = {c.text: c.display_meta_text for c in completer.get_completions(Document("/mcp"), None)}
    assert any("mcp-status" in text for text in found)
    assert local_plugin_reads == []


def _unloaded_since_the_cached_read(monkeypatch, name):
    """The CLI ran ``/name`` and cached the catalog; then a gateway plugin reload dropped it (same
    session, same gateway life), so ``slash.exec`` refuses it as the gateway does."""
    from hermes_cli.gateway_client import GatewayRPCError
    conn = _Conn({name: "Before the reload"})
    c = _cli(monkeypatch, conn)
    c._process_unregistered_slash(f"/{name}", f"/{name}")
    conn._plugins, conn._reply = {}, GatewayRPCError("unsupported_command")
    return conn, c


def test_a_command_a_reload_unloaded_is_named_not_the_raw_refusal(monkeypatch, capsys, local_plugin_reads):
    conn, c = _unloaded_since_the_cached_read(monkeypatch, "gone-plugin")
    capsys.readouterr()
    assert c._process_unregistered_slash("/gone-plugin", "/gone-plugin") is True
    out = capsys.readouterr().out
    assert "no longer has this plugin command" in out and "unsupported_command" not in out, out
    assert _execs(conn) == ["gone-plugin"] * 2 and _catalog_reads(conn) == 2  # one re-read, no retry
    expanded = []
    monkeypatch.setattr(HermesCLI, "_expand_slash_prefix", lambda self, *a: expanded.append(a) or True)
    c._process_unregistered_slash("/gone-plugin", "/gone-plugin")  # the cache is fresh: never sent
    assert _execs(conn) == ["gone-plugin"] * 2 and len(expanded) == 1 and local_plugin_reads == []


def test_a_same_named_skill_runs_once_a_reload_unloaded_the_plugin_command(monkeypatch, capsys):
    conn, c = _unloaded_since_the_cached_read(monkeypatch, "pptwin")
    ran = []
    monkeypatch.setattr(cli_mod, "_ensure_skill_commands", lambda: {"/pptwin": {"name": "pptwin"}})
    monkeypatch.setattr(HermesCLI, "_run_skill_slash_command", lambda self, *a: ran.append(a[0]))
    capsys.readouterr()
    for _ in range(2):
        assert c._process_unregistered_slash("/pptwin", "/pptwin") is True
    assert ran == ["/pptwin", "/pptwin"] and "unsupported_command" not in capsys.readouterr().out
    assert _execs(conn) == ["pptwin"] * 2  # the first run and the one refusal; the skill follows


def test_a_skill_reply_for_an_unloaded_plugin_command_runs_the_skill_here(monkeypatch, capsys):
    """The gateway answers the dropped name for its own same-named skill (``type: skill``): the
    re-read shows no plugin command, so the CLI's skill dispatch runs it, as for a refusal."""
    conn, c = _unloaded_since_the_cached_read(monkeypatch, "pptwin")
    conn._reply = {"type": "skill", "name": "pptwin", "message": "skill body"}
    ran = []
    monkeypatch.setattr(cli_mod, "_ensure_skill_commands", lambda: {"/pptwin": {"name": "pptwin"}})
    monkeypatch.setattr(HermesCLI, "_run_skill_slash_command", lambda self, *a: ran.append(a[0]))
    assert c._process_unregistered_slash("/pptwin", "/pptwin") is True
    assert ran == ["/pptwin"] and "nothing was run" not in capsys.readouterr().out


def test_the_open_sessions_catalog_is_read_and_an_older_gateway_is_asked_without_it(monkeypatch, local_plugin_reads):
    """On its open gateway session the CLI reads that session's catalog (skills from the session's folder);
    a gateway older than this CLI refuses ``session_id`` and is asked again without it."""
    from hermes_cli import gateway_plugin_commands
    from hermes_cli.gateway_client import GatewayRPCError
    conn = _Conn({"mcp-status": "Ask the MCP server"})
    c = _cli(monkeypatch, conn)
    c._gateway_session_open = True
    assert gateway_plugin_commands.lookup(c, "mcp-status") is True
    assert conn.calls[-1] == ("commands.catalog", {"session_id": "local-s"})
    newer = conn.call_tagged

    def older(method, **params):
        if method == "commands.catalog" and params:
            conn.calls.append((method, params))
            raise GatewayRPCError("invalid_params", code=-32602)
        return newer(method, **params)
    monkeypatch.setattr(conn, "call_tagged", older)
    c.session_id = "local-t"  # another session: its catalog is read again
    assert gateway_plugin_commands.lookup(c, "mcp-status") is True
    assert [p for m, p in conn.calls if m == "commands.catalog"][-2:] == [{"session_id": "local-t"}, {}]
    assert local_plugin_reads == []
