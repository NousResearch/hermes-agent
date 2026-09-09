"""AWS Bedrock wizards: native Converse API (IAM chain, region-scoped model discovery)
and the Bedrock API Key mode on the OpenAI-compatible bedrock-mantle endpoint.

Imports of hermes_cli.auth / config / models stay lazy (tests patch them at call time).
Prompt strings and config write order are behavior.
"""

from __future__ import annotations

from hermes_cli.model_setup_flows_common import (
    _ask, _ensure_dict_section, _finish_model, _pick_model_or_prompt, _say)


# AWS cross-region inference profile prefixes. A geo-prefixed profile only routes
# from endpoints in its own geography (us.* from eu-central-2 is rejected by AWS
# regardless of credentials); global.* routes from everywhere.
BEDROCK_GEO_PREFIXES = ("us.", "eu.", "ap.", "apac.", "jp.", "ca.", "sa.", "me.", "af.")

# region-name prefixes -> inference-profile geo prefix
_REGION_GEO = (("us.", ("us-", "us_gov")), ("eu.", ("eu-",)), ("ap.", ("ap-",)), ("ca.", ("ca-",)),
               ("sa.", ("sa-",)), ("me.", ("me-",)), ("af.", ("af-",)))


def bedrock_region_geo_prefix(region_name: str) -> str:
    """Map an AWS region name to its inference-profile geo prefix ('' = unknown)."""
    r = (region_name or "").lower()
    return next((geo for geo, prefixes in _REGION_GEO if r.startswith(prefixes)), "")


def bedrock_model_routable_from_region(model_id: str, region_name: str) -> bool:
    """True when *model_id* can be invoked from *region_name*'s endpoint: bare foundation-model ids
    and ``global.*`` profiles route from anywhere, geo-prefixed profiles only from their own
    geography. Unknown regions hide nothing."""
    mid = (model_id or "").lower()
    matched_geo = next((p for p in BEDROCK_GEO_PREFIXES if mid.startswith(p)), None)
    if matched_geo is None or mid.startswith("global."):
        return True
    geo = bedrock_region_geo_prefix(region_name)
    if not geo:
        return True
    if geo == "ap.":
        # Asia-Pacific regions can carry ap./apac./jp. profile spellings.
        return matched_geo in ("ap.", "apac.", "jp.")
    return matched_geo == geo


def _model_flow_bedrock_api_key(config, region, current_model=""):
    """Bedrock API Key mode on the OpenAI-compatible bedrock-mantle endpoint — for developers
    without an AWS account who received a Bedrock API Key from their AWS admin."""
    from hermes_cli.auth import _resolve_api_key_provider_secret, ProviderConfig
    from hermes_cli.config import save_env_value
    from hermes_cli.models import _PROVIDER_MODELS
    mantle_base_url = f"https://bedrock-mantle.{region}.api.aws/v1"

    # Check env var and credential pool (keys added via `hermes auth`)
    bedrock_pconfig = ProviderConfig(id="bedrock", name="Bedrock", auth_type="api_key", api_key_env_vars=("AWS_BEARER_TOKEN_BEDROCK",))
    existing_key, existing_source = _resolve_api_key_provider_secret("bedrock", bedrock_pconfig)
    if existing_key:
        from hermes_cli.env_loader import format_secret_source_suffix
        source_suffix = format_secret_source_suffix(existing_source or "AWS_BEARER_TOKEN_BEDROCK")
        print(f"  Bedrock API Key: {existing_key[:12]}... ✓{source_suffix}")
    else:
        _say(f"  Endpoint: {mantle_base_url}", "")
        api_key = _ask("  Bedrock API Key: ", secret=True, cancel_msg="")
        if api_key is None:
            return
        if not api_key:
            print("  Cancelled.")
            return
        save_env_value("AWS_BEARER_TOKEN_BEDROCK", api_key)
        existing_key = api_key
        print("  ✓ API key saved.")
    print()

    # Static list — mantle doesn't need boto3 for discovery
    model_list = _PROVIDER_MODELS.get("bedrock", [])
    print(f"  Showing {len(model_list)} curated models")
    selected = _pick_model_or_prompt(
        model_list, "  Model ID: ", current_model=current_model, confirm_provider="custom",
        confirm_base_url=mantle_base_url, confirm_api_key=existing_key)

    def _finish(cfg, _model):
        # The bearer token rides on a named provider entry: a bare ``provider: custom``
        # cannot carry a credential for this host because OPENAI_API_KEY is gated to
        # openai.com, so requests would go out as "no-key-required".
        providers = _ensure_dict_section(cfg, "providers")
        mantle_entry = providers.get("bedrock-mantle")
        if not isinstance(mantle_entry, dict):
            mantle_entry = {}
        mantle_entry["base_url"] = mantle_base_url
        mantle_entry["key_env"] = "AWS_BEARER_TOKEN_BEDROCK"
        providers["bedrock-mantle"] = mantle_entry
        # Also save region in bedrock config for reference
        _ensure_dict_section(cfg, "bedrock")["region"] = region

    # Saved as a custom provider pointing to bedrock-mantle (no inline endpoint fields).
    if _finish_model(selected, "custom:bedrock-mantle", f"  Default model set to: {selected} (via Bedrock API Key, {region})",
                     no_change="  No change.", drop_base_url=True, drop_api_mode=True, finish=_finish) is not None:
        print(f"  Endpoint: {mantle_base_url}")


_BEDROCK_EXCLUDE_PREFIXES = ("stability.", "cohere.embed", "twelvelabs.", "us.stability.", "us.cohere.embed",
                             "us.twelvelabs.", "global.cohere.embed", "global.twelvelabs.")

_BEDROCK_EXCLUDE_SUBSTRINGS = ("safeguard", "voxtral", "palmyra-vision")

_BEDROCK_PROFILE_PREFIXES = BEDROCK_GEO_PREFIXES + ("global.",)

# Recommended models, matched geo-agnostically so an EU (eu.*) or APAC (apac.*)
# picker pins its own region's profile rather than a us.* one.
_BEDROCK_RECOMMENDED_BASES = (
    "anthropic.claude-sonnet-4-6", "anthropic.claude-opus-4-6", "anthropic.claude-haiku-4-5", "amazon.nova-pro",
    "amazon.nova-lite", "amazon.nova-micro", "deepseek.v3", "meta.llama4-maverick", "meta.llama4-scout")


def _bedrock_text_model_ids(live_models: list, region: str) -> list[str]:
    """Filter live Bedrock models to routable text models, dedupe bare ids against their
    inference profiles, and order: recommended (in-region profile before global.*),
    then other global.* profiles, then the rest."""
    def _base_id(mid: str) -> str:
        _pp = next((p for p in _BEDROCK_PROFILE_PREFIXES if mid.startswith(p)), None)
        return mid[len(_pp):] if _pp else mid

    filtered = [
        m for m in live_models
        if not any(m["id"].startswith(p) for p in _BEDROCK_EXCLUDE_PREFIXES)
        and not any(s in m["id"].lower() for s in _BEDROCK_EXCLUDE_SUBSTRINGS)
        and bedrock_model_routable_from_region(m["id"], region)]
    # Deduplicate: prefer inference profiles (geo-prefixed or global.*) over bare foundation model IDs.
    profile_base_ids = {_base_id(m["id"]) for m in filtered if m["id"].startswith(_BEDROCK_PROFILE_PREFIXES)}
    deduped = [m for m in filtered if m["id"].startswith(_BEDROCK_PROFILE_PREFIXES) or m["id"] not in profile_base_ids]

    def _sort_key(m):
        mid = m["id"]
        base = _base_id(mid)
        for i, rec in enumerate(_BEDROCK_RECOMMENDED_BASES):
            if base.startswith(rec):
                # In-region geo profile beats global.* for the same model
                return (0, i, 0 if not mid.startswith("global.") else 1, mid)
        if mid.startswith("global."):
            return (1, 0, 0, mid)
        return (2, 0, 0, mid)

    deduped.sort(key=_sort_key)
    return [m["id"] for m in deduped]


def _model_flow_bedrock(config, current_model=""):
    """AWS Bedrock (native Converse API via boto3): verify credentials, pick region, discover models.
    Auth is the AWS SDK default credential chain, so no API key prompt is needed."""
    from hermes_cli.models import _PROVIDER_MODELS

    # 1. Check for AWS credentials
    try:
        from agent.bedrock_adapter import has_aws_credentials, resolve_aws_auth_env_var, resolve_bedrock_region, discover_bedrock_models
    except ImportError:
        _say("  ✗ boto3 is not installed. Install it with:", "    pip install boto3", "")
        return

    if not has_aws_credentials():
        _say("  ⚠ No AWS credentials detected via environment variables.",
             "  Bedrock will use boto3's default credential chain (IMDS, SSO, etc.)", "")
    auth_var = resolve_aws_auth_env_var()
    _say(f"  AWS credentials: {auth_var} ✓" if auth_var else "  AWS credentials: boto3 default chain (instance role / SSO)",
         "")

    # 2. Region selection
    current_region = resolve_bedrock_region()
    region_input = _ask(f"  AWS Region [{current_region}]: ", cancel_msg="")
    if region_input is None:
        return
    region = region_input or current_region

    # 2b. Authentication mode
    _say("  Choose authentication method:", "", "    1. IAM credential chain (recommended)",
         "       Works with EC2 instance roles, SSO, env vars, aws configure", "    2. Bedrock API Key",
         "       Enter your Bedrock API Key directly — also supports",
         "       team scenarios where an admin distributes keys", "")
    auth_choice = _ask("  Choice [1]: ", raw=True, cancel_msg="")
    if auth_choice is None:
        return
    if auth_choice == "2":
        _model_flow_bedrock_api_key(config, region, current_model)
        return

    # 3. Model discovery — try live API first, fall back to static list
    print(f"  Discovering models in {region}...")
    live_models = discover_bedrock_models(region)
    if live_models:
        model_list = _bedrock_text_model_ids(live_models, region)
        print(f"  Found {len(model_list)} text model(s) (filtered from {len(live_models)} total)")
    else:
        model_list = _PROVIDER_MODELS.get("bedrock", [])
        if not model_list:
            print("  No models found. Check IAM permissions for bedrock:ListFoundationModels.")
            return
        print(f"  Using {len(model_list)} curated models (live discovery unavailable)")

    # 4. Model selection
    runtime_url = f"https://bedrock-runtime.{region}.amazonaws.com"
    selected = _pick_model_or_prompt(model_list, "  Model ID: ", current_model=current_model, confirm_provider="bedrock", confirm_base_url=runtime_url)
    # api_mode is dropped: bedrock_converse is auto-detected.
    _finish_model(selected, "bedrock", f"  Default model set to: {selected} (via AWS Bedrock, {region})", no_change="  No change.",
                  base_url=runtime_url, drop_api_mode=True,
                  finish=lambda cfg, _m: _ensure_dict_section(cfg, "bedrock").__setitem__("region", region))
