"""Fresh exec probes for the private managed safe-worker bootstrap contract."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def run_worker(tmp_path, body, mode="safe"):
    home = tmp_path / mode
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text("agent:\n  max_turns: 17\n", encoding="utf-8")
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(home), PROBE_MODE=mode)
    for key in ("HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG", "HERMES_IGNORE_RULES"):
        env.pop(key, None)
    root = Path(__file__).resolve().parents[2]
    script = home / "worker_probe.py"
    script.write_text(f"import sys; sys.path.insert(0, {str(root)!r})\n" + textwrap.dedent(body))
    result = subprocess.run(
        [sys.executable, str(script)], env=env,
        cwd=root, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mode", ["safe", "config", "ordinary"])
def test_worker_config_is_frozen_before_any_profile_read(tmp_path, mode):
    result = run_worker(tmp_path, '''
        import json, os, sys, threading
        from pathlib import Path
        mode = os.environ["PROBE_MODE"]
        if mode != "ordinary":
            from agent.safe_worker_policy import _bind_safe_worker_policy
            snapshot = {"agent": {"max_turns": 3}, "terminal": {"cwd": "/explicit"}}
            _bind_safe_worker_policy(safe_mode=mode == "safe", ignore_user_config=True, config=snapshot)
            snapshot["agent"]["max_turns"] = 999
        reads = []
        def audit(event, args):
            if event == "open" and str(args[0]).endswith("config.yaml") and args[1] != "w":
                reads.append(str(args[0]))
        sys.addaudithook(audit)
        from hermes_cli.config import load_config, load_config_readonly, read_raw_config, read_raw_config_readonly, DEFAULT_CONFIG
        from hermes_cli import env_loader
        callbacks = []
        env_loader._apply_managed_env = lambda: callbacks.append("managed")
        env_loader._apply_external_secret_sources = lambda home: callbacks.append("secrets")
        env_loader._reapply_terminal_config_bridge = lambda home: callbacks.append("terminal")
        env_loader.load_hermes_dotenv()
        cfg = load_config()
        assert cfg["compression"] == DEFAULT_CONFIG["compression"]
        first = cfg["agent"]["max_turns"]
        cfg["agent"]["max_turns"] = 123
        if mode != "ordinary":
            Path(os.environ["HERMES_HOME"], "config.yaml").write_text("[broken YAML")
            # Audit excludes this fixture write, not the preceding reads.
        results = []
        def check():
            for loader in (load_config, load_config_readonly, read_raw_config, read_raw_config_readonly):
                results.append(loader()["agent"]["max_turns"])
        t = threading.Thread(target=check); t.start(); t.join()
        print(json.dumps({"first": first, "values": results, "reads": reads, "callbacks": callbacks}))
    ''', mode)
    if mode == "ordinary":
        assert result["first"] == 17 and result["reads"]
        assert result["callbacks"] == ["secrets", "managed", "terminal"]
    else:
        assert result == {"first": 3, "values": [3, 3, 3, 3], "reads": [], "callbacks": []}


def test_private_worker_binding_is_one_shot_typed_and_not_env_selected(tmp_path):
    result = run_worker(tmp_path, '''
        import json, os
        from dataclasses import FrozenInstanceError
        from agent.safe_worker_policy import _bind_safe_worker_policy, safe_worker_enabled, worker_config_snapshot
        os.environ["HERMES_SAFE_MODE"] = "1"
        assert not safe_worker_enabled()
        for value in (1, "true", None):
            try:
                _bind_safe_worker_policy(safe_mode=value, ignore_user_config=True, config={})
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError("non-boolean accepted")
        policy = _bind_safe_worker_policy(safe_mode=True, ignore_user_config=False, config={})
        assert policy.ignore_user_config and safe_worker_enabled()
        try:
            policy.safe_mode = False
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError("mutable policy")
        try:
            _bind_safe_worker_policy(safe_mode=False, ignore_user_config=True, config={})
        except RuntimeError:
            pass
        else:
            raise AssertionError("rebound policy")
        assert worker_config_snapshot() == {}
        print(json.dumps({"bound": safe_worker_enabled()}))
    ''')
    assert result == {"bound": True}


@pytest.mark.parametrize("mode", ["safe", "config", "ordinary"])
def test_safe_worker_never_discovers_or_invokes_customizations(tmp_path, mode):
    result = run_worker(tmp_path, r"""
        import json, os, sys
        from pathlib import Path
        mode = os.environ["PROBE_MODE"]
        home = Path(os.environ["HERMES_HOME"])
        (home / "config.yaml").write_text("plugins:\n  enabled: [sentinel]\n")
        events = home / "events"
        def mark(value):
            with events.open("a") as f: f.write(value + "\n")
        for name, kind in (("sentinel", "standalone"), ("model-providers/nested", "model-provider"), ("flat", "model-provider")):
            p = home / "plugins" / name
            p.mkdir(parents=True)
            (p / "plugin.yaml").write_text("name: " + name.split("/")[-1] + "\nversion: 1.0.0\nkind: " + kind + "\n")
            prefix = "from pathlib import Path\nimport os\ndef mark(s):\n    with (Path(os.environ['HERMES_HOME']) / 'events').open('a') as f: f.write(s + '\\n')\n"
            if kind == "standalone":
                code = "mark('plugin-import')\ndef register(ctx):\n    mark('plugin-register')\n    ctx.register_hook('on_session_start', lambda **kw: mark('plugin-hook'))\n"
            else:
                code = "from providers import register_provider\nfrom providers.base import ProviderProfile\nmark('" + name + "')\nregister_provider(ProviderProfile(name='" + name.split("/")[-1] + "'))\n"
            (p / "__init__.py").write_text(prefix + code)
        if mode != "ordinary":
            from agent.safe_worker_policy import _bind_safe_worker_policy
            _bind_safe_worker_policy(safe_mode=mode == "safe", ignore_user_config=True, config={"plugins": {"enabled": ["sentinel"]}})
        reads = []
        def audit(event, args):
            if event == "open" and "/plugins/" in str(args[0]) and args[1] != "w":
                reads.append(str(args[0]))
        sys.addaudithook(audit)
        import providers
        from hermes_cli import plugins, lifecycle
        plugins.discover_plugins()
        providers.list_providers()
        from providers.base import ProviderProfile
        providers.register_provider(ProviderProfile(name="injected"))
        injected = providers.get_provider_profile("injected")
        manager = plugins.get_plugin_manager()
        manager._hooks.setdefault("on_session_start", []).append(lambda **kw: mark("direct-hook"))
        manager._middleware.setdefault("probe", []).append(lambda **kw: mark("middleware"))
        manager.invoke_hook("on_session_start")
        manager.invoke_middleware("probe")
        lifecycle._observe = lambda *a, **kw: mark("observer")
        lifecycle.invoke_hook("on_session_start")
        from agent import shell_hooks, outbound_webhooks
        shell_hooks._spawn = lambda *a: (mark("shell") or {"returncode": 0, "stdout": "", "stderr": "", "error": None, "timed_out": False})
        outbound_webhooks._enqueue = lambda *a: mark("outbound")
        shell_hooks._make_callback(shell_hooks.ShellHookSpec(event="on_session_start", command="inert"))()
        outbound_webhooks._make_callback("on_session_start", outbound_webhooks.WebhookTarget(url="http://127.0.0.1/inert", events=["on_session_start"]))()
        if mode == "safe": assert not reads
        print(json.dumps({"events": events.read_text().splitlines() if events.exists() else [], "reads": reads, "injected": injected is not None}))
    """, mode)
    if mode == "safe":
        assert result == {"events": [], "reads": [], "injected": False}
    else:
        assert {"plugin-import", "plugin-register", "plugin-hook", "model-providers/nested", "flat", "direct-hook", "middleware", "observer", "shell", "outbound"} <= set(result["events"])
        assert result["reads"] and result["injected"]


@pytest.mark.parametrize("mode", ["safe", "config", "ordinary"])
def test_safe_worker_bypasses_rule_readers_even_without_cli_flags(tmp_path, mode):
    result = run_worker(tmp_path, r"""
        import json, os, sys
        from pathlib import Path
        mode = os.environ["PROBE_MODE"]
        home = Path(os.environ["HERMES_HOME"])
        project = home / "project"; project.mkdir()
        (project / "AGENTS.md").write_text("PROJECT_SENTINEL")
        sub = project / "sub"; sub.mkdir()
        (sub / "AGENTS.md").write_text("SUB_SENTINEL")
        (home / "SOUL.md").write_text("SOUL_SENTINEL")
        mem = home / "memories"; mem.mkdir()
        (mem / "MEMORY.md").write_text("MEMORY_SENTINEL")
        skill = home / "skills" / "probe"; skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: probe\ndescription: SKILL_SENTINEL\n---\nSKILL_SENTINEL\n")
        if mode != "ordinary":
            from agent.safe_worker_policy import _bind_safe_worker_policy
            _bind_safe_worker_policy(safe_mode=mode == "safe", ignore_user_config=True, config={})
        reads = []
        selected = {"SOUL.md", "AGENTS.md", "MEMORY.md", "USER.md", "SKILL.md"}
        def audit(event, args):
            if event == "open" and Path(str(args[0])).name in selected and args[1] != "w":
                reads.append(Path(str(args[0])).name)
        sys.addaudithook(audit)
        from agent.prompt_builder import build_context_files_prompt, load_soul_md, build_skills_system_prompt
        from agent.skill_commands import build_preloaded_skills_prompt
        from agent.subdirectory_hints import SubdirectoryHintTracker
        context = build_context_files_prompt(cwd=str(project))
        soul = load_soul_md()
        skills = build_skills_system_prompt()
        preloaded = build_preloaded_skills_prompt(["probe"])
        hints = SubdirectoryHintTracker(str(project)).check_tool_call("read_file", {"path": str(sub / "file.py")})
        from agent.agent_init import _init_memory, _select_context_engine
        from types import SimpleNamespace
        agent = SimpleNamespace(enabled_toolsets=["memory"], disabled_toolsets=[])
        _init_memory(agent, {"memory": {"memory_enabled": True, "user_profile_enabled": True}}, False, "cli")
        # Explicit custom-engine selection must not escape the process policy.
        import plugins.context_engine
        engines = []
        plugins.context_engine.load_context_engine = lambda name: engines.append(name)
        _select_context_engine({"context": {"engine": "sentinel"}})
        print(json.dumps({"context": context, "soul": soul, "skills": skills,
            "preloaded": preloaded, "hints": hints, "reads": reads, "engines": engines,
            "memory": agent._memory_store is not None}))
    """, mode)
    if mode == "safe":
        assert result == {"context": "", "soul": None, "skills": "", "preloaded": ["", [], ["probe"]],
                          "hints": None, "reads": [], "engines": [], "memory": False}
    else:
        assert "PROJECT_SENTINEL" in result["context"] and "SOUL_SENTINEL" in result["soul"]
        assert "SKILL_SENTINEL" in result["skills"] and "SKILL_SENTINEL" in result["preloaded"][0]
        assert "SUB_SENTINEL" in result["hints"]
        assert {"AGENTS.md", "SOUL.md", "SKILL.md", "MEMORY.md"} <= set(result["reads"])
        assert result["engines"] == ["sentinel"] and result["memory"]


@pytest.mark.parametrize("mode", ["safe", "config", "ordinary"])
def test_bare_agent_constructor_cannot_restore_safe_customizations(tmp_path, mode):
    result = run_worker(tmp_path, r"""
        import json, os, sys
        from pathlib import Path
        mode = os.environ["PROBE_MODE"]
        home = Path(os.environ["HERMES_HOME"])
        (home / "SOUL.md").write_text("IDENTITY_SENTINEL")
        if mode != "ordinary":
            from agent.safe_worker_policy import _bind_safe_worker_policy
            _bind_safe_worker_policy(safe_mode=mode == "safe", ignore_user_config=True,
                config={"agent": {"environment_probe": False}})
        reads = []
        def audit(event, args):
            if event == "open" and str(args[0]).endswith("config.yaml") and args[1] != "w":
                reads.append(str(args[0]))
        sys.addaudithook(audit)
        from run_agent import AIAgent
        agent = AIAgent(model="safe-fixture", provider="custom", base_url="http://127.0.0.1:9/v1",
            api_key="fixture", enabled_toolsets=[], quiet_mode=True, skip_context_files=False,
            load_soul_identity=True, skip_background_review=False,
            prefill_messages=[{"role": "user", "content": "PREFILL_SENTINEL"}],
            save_trajectories=False)
        from agent.background_review import load_background_review_settings
        from agent.system_prompt import build_system_prompt
        prompt = build_system_prompt(agent)
        if mode == "safe":
            assert "IDENTITY_SENTINEL" not in prompt
        flags = [agent.skip_context_files, agent.load_soul_identity, agent.skip_background_review]
        print(json.dumps({"flags": flags, "prefill": agent.prefill_messages,
                          "review": load_background_review_settings()[0], "reads": reads}))
    """, mode)
    if mode == "safe":
        assert result == {"flags": [True, False, True], "prefill": [], "review": False, "reads": []}
    else:
        assert result["flags"] == [False, True, False]
        assert result["prefill"] and result["review"]
        if mode == "ordinary":
            assert result["reads"]
        else:
            assert result["reads"] == []


@pytest.mark.parametrize("mode", ["safe", "config", "ordinary"])
def test_direct_secret_hydration_cannot_read_bypassed_yaml(tmp_path, mode):
    result = run_worker(tmp_path, r"""
        import json, os, sys
        from pathlib import Path
        mode = os.environ["PROBE_MODE"]
        home = Path(os.environ["HERMES_HOME"])
        if mode != "ordinary":
            from agent.safe_worker_policy import _bind_safe_worker_policy
            _bind_safe_worker_policy(safe_mode=mode == "safe", ignore_user_config=True, config={})
        reads = []
        def audit(event, args):
            if event == "open" and str(args[0]).endswith("config.yaml") and args[1] != "w":
                reads.append(str(args[0]))
        sys.addaudithook(audit)
        from hermes_cli.env_loader import hydrate_profile_secret_sources
        assert hydrate_profile_secret_sources(home) == {}
        other = home / "other"; other.mkdir()
        (other / "config.yaml").write_text("agent: {}")
        assert hydrate_profile_secret_sources(other) == {}
        print(json.dumps({"reads": reads}))
    """, mode)
    assert bool(result["reads"]) == (mode == "ordinary")


@pytest.mark.linux_only
def test_private_policy_refuses_late_binding_and_fork_inheritance(tmp_path):
    late = run_worker(tmp_path, r"""
        import json
        import hermes_cli.config
        from agent.safe_worker_policy import _bind_safe_worker_policy
        try:
            _bind_safe_worker_policy(safe_mode=True, ignore_user_config=True, config={})
        except RuntimeError as exc:
            assert "before runtime imports" in str(exc)
        else:
            raise AssertionError("late binding accepted")
        print(json.dumps({"refused": True}))
    """, "late")
    assert late == {"refused": True}
    fork = run_worker(tmp_path, r"""
        import json, os
        from agent.safe_worker_policy import _bind_safe_worker_policy, safe_worker_enabled
        _bind_safe_worker_policy(safe_mode=True, ignore_user_config=True, config={})
        pid = os.fork()
        if pid == 0:
            try:
                safe_worker_enabled()
            except RuntimeError as exc:
                os._exit(0 if "fresh exec" in str(exc) else 2)
            os._exit(3)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert safe_worker_enabled()
        print(json.dumps({"refused": True}))
    """, "fork")
    assert fork == {"refused": True}
