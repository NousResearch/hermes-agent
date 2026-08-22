"""Lift #88796 quarantine from one manager object to one process/profile scope."""

from pathlib import Path


manager_path = Path("agent/memory_manager.py")
text = manager_path.read_text(encoding="utf-8")

globals_anchor = '''    verdict: Optional[str] = None


def normalize_tool_schema'''
globals_replacement = '''    verdict: Optional[str] = None


# Gateway agents are reconstructed per message, so quarantine and pending-call
# admission must outlive any one MemoryManager instance. These registries are
# process-local by design: an explicit process restart is the only automatic
# trust reset. Keys contain provider name plus non-secret profile scope only.
_PROCESS_EXTERNAL_PREFETCH_LOCK = threading.RLock()
_PROCESS_EXTERNAL_PREFETCH_CALLS: Dict[str, _ExternalPrefetchCall] = {}
_PROCESS_EXTERNAL_PREFETCH_QUARANTINE: Dict[str, str] = {}


def normalize_tool_schema'''
if text.count(globals_anchor) != 1:
    raise SystemExit(f"process-registry insertion anchor drifted: {text.count(globals_anchor)}")
text = text.replace(globals_anchor, globals_replacement, 1)

init_anchor = '''        self._external_prefetch_calls: Dict[int, _ExternalPrefetchCall] = {}
        self._external_prefetch_quarantine: Dict[int, str] = {}
        self._external_prefetch_lock = threading.RLock()
        self._external_provider_gate_logged: set[tuple[int, str, str]] = set()
'''
init_replacement = '''        # Aliases intentionally point every manager at one process admission
        # table. A gateway message creates a fresh manager; local dictionaries
        # would silently reopen a provider on the next turn.
        self._external_prefetch_calls = _PROCESS_EXTERNAL_PREFETCH_CALLS
        self._external_prefetch_quarantine = _PROCESS_EXTERNAL_PREFETCH_QUARANTINE
        self._external_prefetch_lock = _PROCESS_EXTERNAL_PREFETCH_LOCK
        self._external_provider_scope: Optional[str] = None
        self._external_provider_gate_logged: set[tuple[str, str, str]] = set()
'''
if text.count(init_anchor) != 1:
    raise SystemExit(f"process-registry init anchor drifted: {text.count(init_anchor)}")
text = text.replace(init_anchor, init_replacement, 1)

key_anchor = '''    @staticmethod
    def _external_provider_key(provider: MemoryProvider) -> int:
        """Process-local identity for one registered provider object."""
        return id(provider)
'''
key_replacement = '''    def _external_provider_key(self, provider: MemoryProvider) -> str:
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
if text.count(key_anchor) != 1:
    raise SystemExit(f"provider-key anchor drifted: {text.count(key_anchor)}")
text = text.replace(key_anchor, key_replacement, 1)

initialize_anchor = '''            kwargs["hermes_home"] = str(get_hermes_home())
        for provider in self._providers:
            if not self._provider_call_allowed(provider, "initialize"):
'''
initialize_replacement = '''            kwargs["hermes_home"] = str(get_hermes_home())
        self._external_provider_scope = json.dumps(
            {
                "hermes_home": str(kwargs.get("hermes_home") or "").strip(),
                "agent_identity": str(kwargs.get("agent_identity") or "").strip(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for provider in self._providers:
            if not self._provider_call_allowed(provider, "initialize"):
'''
if text.count(initialize_anchor) != 1:
    raise SystemExit(f"initialize scope anchor drifted: {text.count(initialize_anchor)}")
text = text.replace(initialize_anchor, initialize_replacement, 1)

text = text.replace(
    "quarantine is monotonic for this MemoryManager lifetime.",
    "quarantine is monotonic for the interpreter lifetime in this provider/profile scope.",
)
text = text.replace(
    "permanently quarantined for this manager lifetime",
    "permanently quarantined for this process/profile scope",
)
manager_path.write_text(text, encoding="utf-8")

tests_path = Path("tests/agent/test_memory_async_sync.py")
tests = tests_path.read_text(encoding="utf-8")
tests += r'''


def _initialize_probe_scope(manager, provider, scope) -> None:
    manager.add_provider(provider)
    manager.initialize_all(
        "session-init",
        hermes_home=str(scope),
        agent_identity="default",
    )


def test_quarantine_survives_gateway_manager_recreation(tmp_path, caplog):
    scope = tmp_path / "profile-home"
    first = MemoryManager(external_prefetch_timeout=0.03)
    first_provider = _QuarantineProbeProvider()
    _initialize_probe_scope(first, first_provider, scope)

    _quarantine_provider(first, first_provider, caplog)
    first_provider.prefetch_release.set()

    second = MemoryManager(external_prefetch_timeout=0.03)
    second_provider = _QuarantineProbeProvider()
    second.add_provider(second_provider)
    second.initialize_all(
        "session-next-message",
        hermes_home=str(scope),
        agent_identity="default",
    )

    assert second_provider.calls["initialize"] == 0
    assert second.prefetch_all("new turn secret") == ""
    assert second_provider.calls["prefetch"] == 0
    assert second_provider.name in second.external_provider_quarantine_state

    first.shutdown_all()
    second.shutdown_all()


def test_pending_finalization_fence_spans_concurrent_managers(tmp_path):
    scope = tmp_path / "shared-profile-home"
    first = MemoryManager(external_prefetch_timeout=1.0)
    first_provider = _QuarantineProbeProvider()
    _initialize_probe_scope(first, first_provider, scope)

    second = MemoryManager(external_prefetch_timeout=1.0)
    second_provider = _QuarantineProbeProvider()
    _initialize_probe_scope(second, second_provider, scope)

    result_box = {}

    def run_first():
        result_box["value"] = first._prefetch_provider(
            first_provider,
            "first sensitive query",
            session_id="one",
        )

    worker = threading.Thread(target=run_first)
    worker.start()
    assert first_provider.prefetch_started.wait(timeout=1)

    assert second.prefetch_all("concurrent secret", session_id="two") == ""
    assert second_provider.calls["prefetch"] == 0

    first_provider.prefetch_release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert result_box["value"] == "late context"

    first.shutdown_all()
    second.shutdown_all()
'''
tests_path.write_text(tests, encoding="utf-8")
