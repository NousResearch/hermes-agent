"""Wizard flows (``hermes model``) must honour declared ``providers.<slug>.models``.

A ``providers.<slug>.models: [...]`` block extends that provider's model list on every
picker row surface (``list_authenticated_providers`` sections 1/2/2b and lmstudio);
the ``hermes model`` wizard flows must extend the list they prompt with the same way —
declared-first and deduped. Offline and deterministic: the credential step, the catalog
source, pricing, and the selection prompt are all stubbed — no network, no config writes.
"""


def test_openrouter_flow_prompts_with_declared_ids(monkeypatch):
    """``_model_flow_openrouter`` prompts with the declared ids merged in.

    The flow's catalog does not contain the declared id, so the pre-fix list silently
    drops it. The second arm declares an id the catalog ALSO carries, proving the merge
    dedupes it to a single declared-first entry instead of repeating it.
    """
    import hermes_cli.auth as auth_module
    import hermes_cli.models as hm
    from hermes_cli import model_setup_flows as setup

    declared = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    curated = ["z-ai/glm-5.2", "openai/gpt-5.4"]
    captured: list = []

    # The flow does function-level imports of model_ids / _prompt_model_selection, so
    # patching the source-module attributes binds what the call resolves at run time.
    monkeypatch.setattr(setup, "_ensure_flow_api_key", lambda *_a, **_kw: ("sk-test", "sk-test", False))
    monkeypatch.setattr(setup, "_finish_model", lambda *_a, **_kw: None)
    monkeypatch.setattr(hm, "model_ids", lambda **_kw: list(curated))
    monkeypatch.setattr("hermes_cli.models_pricing.get_pricing_for_provider", lambda *_a, **_kw: {})
    monkeypatch.setattr(
        auth_module, "_prompt_model_selection",
        lambda model_ids, **_kw: captured.append(list(model_ids)) or "")

    config = {"providers": {"openrouter": {"models": [declared]}}}
    setup._model_flow_openrouter(config)
    assert captured[-1] == [declared, *curated]

    # Declared id also present in the catalog: exactly one entry, declared-first.
    monkeypatch.setattr(hm, "model_ids", lambda **_kw: [declared, *curated])
    setup._model_flow_openrouter(config)
    assert captured[-1] == [declared, *curated]


def test_xai_oauth_flow_prompts_with_declared_ids(monkeypatch):
    """``_model_flow_xai_oauth`` prompts with the declared ids merged in.

    The OAuth gate, the catalog and the selection prompt are stubbed; the flow's
    own catalog has no ``xai-oauth`` entry for the declared id, so the pre-fix
    list silently drops it. Second arm: declared id already in the catalog —
    merged exactly once, declared-first.
    """
    import hermes_cli.auth as auth_module
    import hermes_cli.models as hm
    from hermes_cli import model_setup_flows as setup

    declared = "grok-4.6-heavy"
    curated = ["grok-4.6", "grok-4.1-fast"]
    captured: list = []

    # The flow does function-level imports from hermes_cli.auth / hermes_cli.models,
    # so patching the source-module attributes binds what the calls resolve at run time.
    monkeypatch.setattr(setup, "_oauth_gate", lambda *_a, **_kw: True)
    monkeypatch.setattr(setup, "_activate_provider_model", lambda *_a, **_kw: None)
    monkeypatch.setattr(auth_module, "get_xai_oauth_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(
        auth_module, "resolve_xai_oauth_runtime_credentials",
        lambda: {"base_url": "https://api.x.ai/v1"})
    monkeypatch.setattr(hm, "provider_model_ids", lambda *_a, **_kw: list(curated))
    monkeypatch.setattr(
        auth_module, "_prompt_model_selection",
        lambda model_ids, **_kw: captured.append(list(model_ids)) or "")

    config = {"providers": {"xai-oauth": {"models": [declared]}}}
    setup._model_flow_xai_oauth(config)
    assert captured[-1] == [declared, *curated]

    # Declared id also present in the catalog: exactly one entry, declared-first.
    monkeypatch.setattr(hm, "provider_model_ids", lambda *_a, **_kw: [declared, *curated])
    setup._model_flow_xai_oauth(config)
    assert captured[-1] == [declared, *curated]


def _stub_bedrock_prompts(monkeypatch, captured: list):
    """Stub the Bedrock wizard's prompt + persist seams; return the captured list."""
    from hermes_cli import model_setup_flows_bedrock as setup_bedrock

    monkeypatch.setattr(
        setup_bedrock, "_pick_model_or_prompt",
        lambda model_list, *_a, **_kw: captured.append(list(model_list)) or "")
    monkeypatch.setattr(setup_bedrock, "_finish_model", lambda *_a, **_kw: None)
    return setup_bedrock


def test_bedrock_flow_prompts_with_declared_ids(monkeypatch):
    """``_model_flow_bedrock`` prompts with declared ids merged over both branches.

    Credentials, region prompts, auth-mode choice and the selection prompt are
    stubbed; live discovery is stubbed with a fixed model set. The declared id is in
    neither branch's list pre-fix, so the wrapped prompt list must carry it
    first. Second arm: declared id already discovered — merged exactly once.
    Third arm: live discovery empty (static curated fallback) — also merged.
    """
    import agent.bedrock_adapter as bedrock_adapter
    import hermes_cli.models as hm

    declared = "custom.bedrock.nova-4"
    live = [{"id": "anthropic.claude-sonnet-4-6"}, {"id": "meta.llama4-maverick"}]
    captured: list = []
    setup_bedrock = _stub_bedrock_prompts(monkeypatch, captured)

    monkeypatch.setattr(bedrock_adapter, "has_aws_credentials", lambda: True)
    monkeypatch.setattr(bedrock_adapter, "resolve_aws_auth_env_var", lambda: "AWS_ACCESS_KEY_ID")
    monkeypatch.setattr(bedrock_adapter, "resolve_bedrock_region", lambda: "us-east-1")
    # Region prompt and auth-mode choice both take the default ("") answer.
    monkeypatch.setattr(setup_bedrock, "_ask", lambda *_a, **_kw: "")

    # Live-discovery branch.
    monkeypatch.setattr(bedrock_adapter, "discover_bedrock_models", lambda _region: list(live))
    setup_bedrock._model_flow_bedrock({"providers": {"bedrock": {"models": [declared]}}})
    assert captured[-1] == [declared, "anthropic.claude-sonnet-4-6", "meta.llama4-maverick"]

    # Declared id already in the discovered set: exactly one entry, declared-first.
    monkeypatch.setattr(bedrock_adapter, "discover_bedrock_models",
                        lambda _region: [{"id": declared}] + list(live))
    setup_bedrock._model_flow_bedrock({"providers": {"bedrock": {"models": [declared]}}})
    assert captured[-1] == [declared, "anthropic.claude-sonnet-4-6", "meta.llama4-maverick"]

    # Static curated fallback branch (discovery unavailable).
    monkeypatch.setattr(bedrock_adapter, "discover_bedrock_models", lambda _region: [])
    monkeypatch.setattr(hm, "_PROVIDER_MODELS", {"bedrock": ["anthropic.claude-sonnet-4-6"]})
    setup_bedrock._model_flow_bedrock({"providers": {"bedrock": {"models": [declared]}}})
    assert captured[-1] == [declared, "anthropic.claude-sonnet-4-6"]

    # No declaration at all: the flow's own list passes through untouched.
    setup_bedrock._model_flow_bedrock({})
    assert captured[-1] == ["anthropic.claude-sonnet-4-6"]


def test_bedrock_api_key_flow_prompts_with_declared_ids(monkeypatch):
    """``_model_flow_bedrock_api_key`` merges declared ids into the mantle static list.

    The declaration key is still ``bedrock`` even though the flow confirms as a
    ``custom`` provider entry (bedrock-mantle). The secret resolution and the
    prompt are stubbed — no key prompt, no config writes.
    """
    import hermes_cli.auth as auth_module
    import hermes_cli.models as hm

    declared = "custom.bedrock.nova-4"
    captured: list = []
    setup_bedrock = _stub_bedrock_prompts(monkeypatch, captured)

    monkeypatch.setattr(auth_module, "_resolve_api_key_provider_secret",
                        lambda *_a, **_kw: ("sk-mantle-test", "AWS_BEARER_TOKEN_BEDROCK"))
    monkeypatch.setattr(hm, "_PROVIDER_MODELS", {"bedrock": ["anthropic.claude-sonnet-4-6"]})

    setup_bedrock._model_flow_bedrock_api_key(
        {"providers": {"bedrock": {"models": [declared]}}}, "us-east-1")
    assert captured[-1] == [declared, "anthropic.claude-sonnet-4-6"]

    # No declaration: curated list passes through untouched.
    setup_bedrock._model_flow_bedrock_api_key({}, "us-east-1")
    assert captured[-1] == ["anthropic.claude-sonnet-4-6"]
