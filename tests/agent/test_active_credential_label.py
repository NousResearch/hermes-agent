"""Opt-in ``account`` status-bar field: the shared label resolver and the TUI usage payload.

The classic CLI bar (``tests/cli/test_cli_status_bar.py``) and the TUI status rule both read
``active_credential_label``; these tests pin the resolver's contract so the two surfaces can
never disagree about which pooled credential a session is dispatching with.
"""

from types import SimpleNamespace

from agent.agent_runtime_helpers import active_credential_label


def _pool(entries, *, current=None, resolved_id=None):
    return SimpleNamespace(
        entries=lambda: list(entries),
        current=lambda: current,
        entry_id_for_api_key=lambda hint=None: resolved_id,
    )


def _entry(entry_id, label, key="sk"):
    return SimpleNamespace(id=entry_id, label=label, runtime_api_key=key)


class TestActiveCredentialLabel:
    def test_no_pool_bound_is_empty(self):
        assert active_credential_label(SimpleNamespace(api_key="sk-env")) == ""
        assert active_credential_label(SimpleNamespace(_credential_pool=None)) == ""

    def test_dispatch_entry_id_wins_over_pool_cursor(self):
        """After a failover the pool cursor may already point elsewhere; the label must
        describe the credential this agent actually dispatches with."""
        mine, other = _entry("a1", "alice@example.com", "sk-a"), _entry("b2", "bob@example.com", "sk-b")
        agent = SimpleNamespace(
            api_key="sk-a", _credential_pool_entry_id="a1",
            _credential_pool=_pool([mine, other], current=other, resolved_id="b2"))
        assert active_credential_label(agent) == "alice@example.com"

    def test_falls_back_to_key_match_when_entry_id_unbound(self):
        mine, other = _entry("a1", "alice@example.com", "sk-a"), _entry("b2", "bob@example.com", "sk-b")
        agent = SimpleNamespace(
            api_key="sk-b", _credential_pool_entry_id=None,
            _credential_pool=_pool([mine, other], current=mine, resolved_id="b2"))
        assert active_credential_label(agent) == "bob@example.com"

    def test_falls_back_to_pool_cursor_when_nothing_resolves(self):
        cur = _entry("c3", "carol@example.com")
        agent = SimpleNamespace(
            api_key=None, _credential_pool_entry_id="gone",
            _credential_pool=_pool([cur], current=cur, resolved_id=None))
        assert active_credential_label(agent) == "carol@example.com"

    def test_label_is_stripped_and_never_token_material(self):
        entry = SimpleNamespace(id="a1", label="  alice@example.com  ", runtime_api_key="sk-secret-token")
        agent = SimpleNamespace(
            api_key="sk-secret-token", _credential_pool_entry_id="a1",
            _credential_pool=_pool([entry], current=entry, resolved_id="a1"))
        label = active_credential_label(agent)
        assert label == "alice@example.com"
        assert "sk-secret" not in label

    def test_pool_errors_never_propagate(self):
        class _Boom:
            def entries(self):
                raise RuntimeError("pool exploded")

            def current(self):
                raise RuntimeError("pool exploded")

            def entry_id_for_api_key(self, hint=None):
                raise RuntimeError("pool exploded")

        agent = SimpleNamespace(api_key="sk", _credential_pool_entry_id="a1", _credential_pool=_Boom())
        assert active_credential_label(agent) == ""


class TestGatewayUsagePayload:
    """``tui_gateway.server._get_usage`` carries the label to the TUI status rule."""

    def _agent(self, pool=None):
        return SimpleNamespace(
            model="m", session_input_tokens=1, session_output_tokens=1, session_prompt_tokens=1,
            session_completion_tokens=1, session_total_tokens=2, session_api_calls=1,
            context_compressor=None, api_key="sk-a", _credential_pool_entry_id="a1",
            _credential_pool=pool)

    def test_account_label_present_when_pool_bound(self):
        from tui_gateway import server

        entry = _entry("a1", "alice@example.com", "sk-a")
        usage = server._get_usage(self._agent(_pool([entry], current=entry, resolved_id="a1")))
        assert usage["account_label"] == "alice@example.com"

    def test_account_label_omitted_not_blanked_without_pool(self):
        """The TUI segment self-hides on a missing key; an empty string would be a lie."""
        from tui_gateway import server

        usage = server._get_usage(self._agent(pool=None))
        assert "account_label" not in usage
