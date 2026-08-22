"""Lift #88796's process quarantine to pre-registration admission scope."""

from __future__ import annotations

from pathlib import Path
import subprocess


manager_path = Path("agent/memory_manager.py")
text = manager_path.read_text(encoding="utf-8")

signature_anchor = (
    "    def __init__(self, *, external_prefetch_timeout: Optional[float] = None) -> None:\n"
)
signature_replacement = '''    def __init__(
        self,
        *,
        external_prefetch_timeout: Optional[float] = None,
        external_provider_scope: Optional[str] = None,
    ) -> None:
'''
if text.count(signature_anchor) != 1:
    raise SystemExit(f"MemoryManager signature drifted: {text.count(signature_anchor)}")
text = text.replace(signature_anchor, signature_replacement, 1)

scope_anchor = '''        self._external_prefetch_lock = _PROCESS_EXTERNAL_PREFETCH_LOCK
        self._external_provider_scope: Optional[str] = None
        self._external_provider_gate_logged: set[tuple[str, str, str]] = set()
'''
scope_replacement = '''        self._external_prefetch_lock = _PROCESS_EXTERNAL_PREFETCH_LOCK
        normalized_scope = str(external_provider_scope or "").strip()
        # Production passes the context-local Hermes home before provider
        # registration. The unique fallback preserves isolation for direct
        # callers/tests that do not participate in gateway process admission.
        self._external_provider_scope = (
            normalized_scope or f"unscoped-manager:{id(self)}"
        )
        self._external_provider_gate_logged: set[tuple[str, str, str]] = set()
'''
if text.count(scope_anchor) != 1:
    raise SystemExit(f"process scope init drifted: {text.count(scope_anchor)}")
text = text.replace(scope_anchor, scope_replacement, 1)

key_anchor = '''    def _external_provider_key(self, provider: MemoryProvider) -> str:
        """Stable process admission key for one provider/profile scope.

        Production managers acquire a canonical non-secret scope during
        ``initialize_all``. The manager-local fallback is used only before
        initialization (principally unit tests) so unrelated uninitialized
        managers cannot poison one another accidentally.
        """
        scope = self._external_provider_scope
        if scope is None:
            scope = f"uninitialized-manager:{id(self)}"
        return f"{str(provider.name).strip().lower()}\\x1f{scope}"
'''
key_replacement = '''    def _external_provider_key(self, provider: MemoryProvider) -> str:
        """Stable process admission key for one provider/profile scope."""
        return (
            f"{str(provider.name).strip().lower()}"
            f"\\x1f{self._external_provider_scope}"
        )
'''
if text.count(key_anchor) != 1:
    raise SystemExit(f"provider key drifted: {text.count(key_anchor)}")
text = text.replace(key_anchor, key_replacement, 1)

late_scope_anchor = '''            kwargs["hermes_home"] = str(get_hermes_home())
        self._external_provider_scope = json.dumps(
            {
                "hermes_home": str(kwargs.get("hermes_home") or "").strip(),
                "agent_identity": str(kwargs.get("agent_identity") or "").strip(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for provider in self._providers:
'''
late_scope_replacement = '''            kwargs["hermes_home"] = str(get_hermes_home())
        for provider in self._providers:
'''
if text.count(late_scope_anchor) != 1:
    raise SystemExit(f"late scope assignment drifted: {text.count(late_scope_anchor)}")
text = text.replace(late_scope_anchor, late_scope_replacement, 1)

registration_anchor = '''        self._providers.append(provider)

        # Core tool names are reserved — a memory provider must never register
'''
registration_replacement = '''        self._providers.append(provider)

        # Registration is a provider call surface: schema discovery can execute
        # plugin code. A process-quarantined provider remains represented for
        # truthful status/shutdown, but receives no registration call and
        # contributes no callable tools.
        if not is_builtin and not self._provider_call_allowed(provider, "register"):
            return

        # Core tool names are reserved — a memory provider must never register
'''
if text.count(registration_anchor) != 1:
    raise SystemExit(f"registration fence anchor drifted: {text.count(registration_anchor)}")
text = text.replace(registration_anchor, registration_replacement, 1)
manager_path.write_text(text, encoding="utf-8")

agent_path = Path("agent/agent_init.py")
agent = agent_path.read_text(encoding="utf-8")
construction_anchor = '''                agent._memory_manager = _MemoryManager()
                _mp = _load_mem(_mem_provider_name)
'''
construction_replacement = '''                # Establish process/profile quarantine authority before
                # add_provider() can execute schema discovery or any other
                # provider registration hook. Gateway agents are rebuilt per
                # message, while this scope remains stable for the process.
                agent._memory_manager = _MemoryManager(
                    external_provider_scope=str(get_hermes_home())
                )
                _mp = _load_mem(_mem_provider_name)
'''
if agent.count(construction_anchor) != 1:
    raise SystemExit(f"agent memory-manager construction drifted: {agent.count(construction_anchor)}")
agent_path.write_text(
    agent.replace(construction_anchor, construction_replacement, 1),
    encoding="utf-8",
)

tests_path = Path("tests/agent/test_memory_async_sync.py")
tests = tests_path.read_text(encoding="utf-8")

replacements = {
    '''    first = MemoryManager(external_prefetch_timeout=0.03)
    first_provider = _QuarantineProbeProvider()
    _initialize_probe_scope(first, first_provider, scope)
''': '''    first = MemoryManager(
        external_prefetch_timeout=0.03,
        external_provider_scope=str(scope),
    )
    first_provider = _QuarantineProbeProvider()
    _initialize_probe_scope(first, first_provider, scope)
''',
    '''    second = MemoryManager(external_prefetch_timeout=0.03)
    second_provider = _QuarantineProbeProvider()
    second.add_provider(second_provider)
''': '''    second = MemoryManager(
        external_prefetch_timeout=0.03,
        external_provider_scope=str(scope),
    )
    second_provider = _QuarantineProbeProvider()
    second.add_provider(second_provider)
''',
    '''    first = MemoryManager(external_prefetch_timeout=1.0)
    first_provider = _QuarantineProbeProvider()
    _initialize_probe_scope(first, first_provider, scope)
''': '''    first = MemoryManager(
        external_prefetch_timeout=1.0,
        external_provider_scope=str(scope),
    )
    first_provider = _QuarantineProbeProvider()
    _initialize_probe_scope(first, first_provider, scope)
''',
    '''    second = MemoryManager(external_prefetch_timeout=1.0)
    second_provider = _QuarantineProbeProvider()
    _initialize_probe_scope(second, second_provider, scope)
''': '''    second = MemoryManager(
        external_prefetch_timeout=1.0,
        external_provider_scope=str(scope),
    )
    second_provider = _QuarantineProbeProvider()
    _initialize_probe_scope(second, second_provider, scope)
''',
}
for old, new in replacements.items():
    if tests.count(old) != 1:
        raise SystemExit(f"scope test anchor drifted: {old!r} -> {tests.count(old)}")
    tests = tests.replace(old, new, 1)

assertion_anchor = '''    assert second_provider.calls["initialize"] == 0
    assert second.prefetch_all("new turn secret") == ""
'''
assertion_replacement = '''    assert second_provider.calls["tool_schema"] == 0
    assert second_provider.calls["initialize"] == 0
    assert second.prefetch_all("new turn secret") == ""
'''
if tests.count(assertion_anchor) != 1:
    raise SystemExit(f"registration assertion anchor drifted: {tests.count(assertion_anchor)}")
tests_path.write_text(
    tests.replace(assertion_anchor, assertion_replacement, 1),
    encoding="utf-8",
)

subprocess.run(
    [
        "python",
        "-m",
        "py_compile",
        "agent/memory_manager.py",
        "agent/agent_init.py",
        "tests/agent/test_memory_async_sync.py",
    ],
    check=True,
)
subprocess.run(
    [
        "python",
        "-m",
        "pytest",
        "-q",
        "tests/agent/test_memory_async_sync.py",
        "tests/run_agent/test_memory_provider_init.py",
    ],
    check=True,
)
subprocess.run(["git", "diff", "--check"], check=True)
subprocess.run(
    [
        "git",
        "add",
        "agent/memory_manager.py",
        "agent/agent_init.py",
        "tests/agent/test_memory_async_sync.py",
    ],
    check=True,
)
subprocess.run(["git", "commit", "--amend", "--no-edit"], check=True)
