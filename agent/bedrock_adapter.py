"""AWS Bedrock Converse API adapter (boto3, optional dependency).

Works without API keys via the AWS credential chain, cross-region inference profiles, guardrails and
control-plane model discovery. OpenAI-format messages/tools are converted to Converse on the way in
and responses normalized back to OpenAI-shaped objects.
"""

import base64
import importlib
import json
import logging
import os
import re
import time
import traceback
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# boto3 is not in the [all] extras; lazy_deps installs it on demand.
try:
    # --------------------------------------------------------------------------- Ensure boto3/botocore are
    # installed before any code in this module runs. Upstream removed boto3 from [all] extras (PRs #24220,
    # #24515); lazy_deps handles on-demand installation so the Bedrock provider still works in the EKS
    # deployment without baking boto3 into the base image.
    # ---------------------------------------------------------------------------
    from tools.lazy_deps import ensure
    ensure("provider.bedrock", prompt=False)
except Exception:
    pass  # let downstream imports surface the real error


_bedrock_runtime_client_cache: Dict[str, Any] = {}
_bedrock_control_client_cache: Dict[str, Any] = {}

# Bedrock-hosted GPT-5.x models are served from the Bedrock Mantle OpenAI-compatible endpoint, not
# Converse. Narrow allowlist so GPT-OSS models stay on the native path.
BEDROCK_OPENAI_RESPONSES_MODEL_IDS: Tuple[str, ...] = (
    "openai.gpt-5.5", "openai.gpt-5.6-sol", "openai.gpt-5.6-terra", "openai.gpt-5.6-luna",
)
_BEDROCK_OPENAI_HOST_RE = re.compile(r"^bedrock-mantle\.([a-z0-9-]+)\.api\.aws$", re.IGNORECASE)
_MIN_BOTO3_VERSION = (1, 34, 59)


def _require_boto3():
    """Import boto3; converse_stream() needs >= 1.34.59 (a system boto3 can shadow the venv pin)."""
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "The 'boto3' package is required for the AWS Bedrock provider. "
            "Install it with: pip install boto3\n"
            "Or install Hermes with Bedrock support: pip install -e '.[bedrock]'"
        )
    try:
        version = tuple(int(x) for x in boto3.__version__.split(".")[:3])
    except (AttributeError, ValueError):
        return boto3  # can't parse — don't block on version check
    if version < _MIN_BOTO3_VERSION:
        raise RuntimeError(
            f"boto3 {boto3.__version__} does not support converse_stream "
            f"(minimum 1.34.59 required). Upgrade with: pip install --upgrade boto3"
        )
    return boto3


def _cached_client(cache: Dict[str, Any], service: str, region: str):
    """Get or create a per-region boto3 client using the default credential chain."""
    if region not in cache:
        cache[region] = _require_boto3().client(service, region_name=region)
    return cache[region]


def _get_bedrock_runtime_client(region: str):
    return _cached_client(_bedrock_runtime_client_cache, "bedrock-runtime", region)


def _get_bedrock_control_client(region: str):
    return _cached_client(_bedrock_control_client_cache, "bedrock", region)


def reset_client_cache():
    """Clear cached boto3 clients. Used in tests and profile switches."""
    _bedrock_runtime_client_cache.clear()
    _bedrock_control_client_cache.clear()


def invalidate_runtime_client(region: str) -> bool:
    """Evict one region's cached ``bedrock-runtime`` client (stale HTTP pool); True if evicted."""
    return _bedrock_runtime_client_cache.pop(region, None) is not None


# --- Bedrock Mantle / OpenAI Responses support ---

def is_openai_bedrock_model(model_id: str) -> bool:
    """True for Bedrock-hosted OpenAI models that require Mantle (GPT-OSS excluded)."""
    return str(model_id or "").strip().lower() in {m.lower() for m in BEDROCK_OPENAI_RESPONSES_MODEL_IDS}


def merge_bedrock_openai_model_ids(model_ids: List[str]) -> List[str]:
    """Append Mantle-only OpenAI models, which control-plane discovery never lists."""
    merged = list(model_ids or [])
    seen = {str(m).lower() for m in merged}
    return merged + [m for m in BEDROCK_OPENAI_RESPONSES_MODEL_IDS if m.lower() not in seen]


def bedrock_openai_base_url(region: str) -> str:
    """Return Bedrock Mantle's OpenAI-compatible base URL for *region*."""
    resolved = (region or "").strip() or resolve_bedrock_runtime_region()
    return f"https://bedrock-mantle.{resolved}.api.aws/openai/v1"


def _mantle_url_parts(base_url: str) -> Tuple[Optional[str], str]:
    """(region or None if not a Mantle host, normalized path) for a base URL."""
    parsed = urlparse(str(base_url or ""))
    match = _BEDROCK_OPENAI_HOST_RE.match(parsed.hostname or "")
    return (match.group(1) if match else None), (parsed.path or "").rstrip("/").lower()


def bedrock_openai_region_from_base_url(base_url: str) -> Optional[str]:
    """Extract the AWS region from a Bedrock Mantle OpenAI base URL."""
    return _mantle_url_parts(base_url)[0]


def is_bedrock_openai_base_url(base_url: str) -> bool:
    """True for Bedrock Mantle endpoints (bare host or /openai[/v1] path)."""
    region, path = _mantle_url_parts(base_url)
    return region is not None and path in {"", "/openai", "/openai/v1"}


def resolve_bedrock_bearer_token(env: Optional[Dict[str, str]] = None) -> str:
    """Return AWS_BEARER_TOKEN_BEDROCK when Bedrock API-key auth is configured."""
    env = env if env is not None else os.environ
    return (env.get("AWS_BEARER_TOKEN_BEDROCK", "") or "").strip()


class BedrockOpenAISigV4Auth(httpx.Auth):
    """httpx auth hook that SigV4-signs Bedrock Mantle OpenAI requests."""

    requires_request_body = True

    def __init__(self, region: str, service: str = "bedrock"):
        self.region = (region or "").strip() or resolve_bedrock_runtime_region()
        self.service = service

    def auth_flow(self, request):  # pragma: no cover - exercised by live call
        import botocore.session
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        credentials = botocore.session.get_session().get_credentials()
        if credentials is None:
            raise RuntimeError(
                "No AWS credentials available for Bedrock OpenAI Responses. "
                "Configure AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, AWS_PROFILE, SSO, or an instance/task role."
            )
        # SigV4 must own Authorization: drop the SDK's placeholder bearer header.
        headers = {str(k): str(v) for k, v in request.headers.items()
                   if str(k).lower() not in {"authorization", "x-amz-date", "x-amz-security-token"}}
        aws_request = AWSRequest(method=request.method, url=str(request.url), data=request.content or b"", headers=headers)
        SigV4Auth(credentials.get_frozen_credentials(), self.service, self.region).add_auth(aws_request)
        request.headers.update(dict(aws_request.headers.items()))
        yield request


def build_bedrock_openai_http_client(region: str, *, timeout: Optional[float] = None):
    """Build an httpx client that SigV4-signs Bedrock OpenAI requests."""
    kwargs: Dict[str, Any] = {"auth": BedrockOpenAISigV4Auth(region)}
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0:
        kwargs["timeout"] = timeout
    return httpx.Client(**kwargs)


def configure_bedrock_openai_client_kwargs(client_kwargs: Dict[str, Any], *, timeout: Optional[float] = None) -> Dict[str, Any]:
    """Install SigV4 auth on OpenAI SDK kwargs for Bedrock Mantle; a real API key keeps the SDK's
    bearer auth, the ``aws-sdk``/``no-key-required`` placeholders mean IAM chain auth."""
    base_url = str(client_kwargs.get("base_url") or "")
    api_key = client_kwargs.get("api_key")
    if not is_bedrock_openai_base_url(base_url) or (
        isinstance(api_key, str) and api_key.strip() and api_key not in {"aws-sdk", "no-key-required"}
    ):
        return client_kwargs
    region = bedrock_openai_region_from_base_url(base_url) or resolve_bedrock_runtime_region()
    client_kwargs["api_key"] = "aws-sdk"
    client_kwargs["http_client"] = build_bedrock_openai_http_client(region, timeout=timeout)
    return client_kwargs


# --- Stale-connection detection ---
# A pooled connection killed under boto3 (NAT timeout, VPN flap, RST) surfaces as a botocore/urllib3
# transport error or a bare AssertionError from urllib3's pool checks; retrying the same client
# reproduces it, so the fix is to evict the client.

_STALE_LIB_MODULE_PREFIXES = ("urllib3.", "botocore.", "boto3.")


def _stale_error_types() -> tuple:
    """botocore + urllib3 transport-failure exception classes (best-effort import)."""
    types: list = []
    for module, names in (
        ("botocore.exceptions", ("ConnectionError", "HTTPClientError")),
        ("urllib3.exceptions", ("ProtocolError", "NewConnectionError", "ConnectionError")),
    ):
        # AttributeError too: ``from mod import Name`` raised ImportError for a missing name.
        with suppress(ImportError, AttributeError):  # pragma: no cover — both present with boto3
            types += [getattr(importlib.import_module(module), name) for name in names]
    return tuple(types)


def is_stale_connection_error(exc: BaseException) -> bool:
    """True for botocore/urllib3 transport errors or AssertionErrors raised inside those libs."""
    if isinstance(exc, _stale_error_types()):
        return True
    return isinstance(exc, AssertionError) and any(
        (frame.f_globals.get("__name__", "") or "").startswith(_STALE_LIB_MODULE_PREFIXES)
        for frame, _lineno in traceback.walk_tb(exc.__traceback__)
    )


def is_streaming_access_denied_error(exc: BaseException) -> bool:
    """True when IAM denied ``bedrock:InvokeModelWithResponseStream`` (permanent: callers fall back
    to converse()). Message-based: the AnthropicBedrock SDK wraps the response but keeps the action name."""
    msg = str(exc).lower()
    if "invokemodelwithresponsestream" not in msg:
        return False
    with suppress(ImportError):  # pragma: no cover — botocore always present with boto3
        from botocore.exceptions import ClientError
        if isinstance(exc, ClientError):
            code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")
            return code in ("AccessDeniedException", "UnauthorizedException")
    return "not authorized" in msg or "accessdenied" in msg


# --- AWS credential detection ---
# Priority order; the first group whose vars are ALL set names the auth source.
_AWS_AUTH_ENV_CHAIN: Tuple[Tuple[str, ...], ...] = (
    ("AWS_BEARER_TOKEN_BEDROCK",),                    # Bedrock bearer token
    ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),   # explicit IAM key pair
    ("AWS_PROFILE",),                                 # named profile (SSO, assume-role)
    ("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",),      # ECS / CodeBuild
    ("AWS_WEB_IDENTITY_TOKEN_FILE",),                 # EKS IRSA
)


def _boto3_chain_has_credentials() -> bool:
    """True if boto3's default chain resolves credentials (IMDS, task role, ...)."""
    with suppress(Exception):
        import botocore.session
        credentials = botocore.session.get_session().get_credentials()
        resolved = credentials.get_frozen_credentials() if credentials is not None else None
        return bool(resolved and resolved.access_key)
    return False


def resolve_aws_auth_env_var(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Name of the active AWS auth source: env vars first (no I/O), then ``"iam-role"`` via boto3's chain, else None."""
    env = env if env is not None else os.environ
    for group in _AWS_AUTH_ENV_CHAIN:
        if all(env.get(var, "").strip() for var in group):
            return group[0]
    return "iam-role" if _boto3_chain_has_credentials() else None


def has_aws_credentials(env: Optional[Dict[str, str]] = None) -> bool:
    """True if any AWS credential source (env vars or boto3 chain) is detected.

    This two-tier approach mirrors the pattern from OpenClaw PR #62673: cloud environments (EC2, ECS,
    Lambda) provide credentials via instance metadata, not environment variables. The env-var check is a
    fast path for local development; the boto3 fallback covers all cloud deployments.
    """
    return resolve_aws_auth_env_var(env) is not None or _boto3_chain_has_credentials()


def resolve_bedrock_region(env: Optional[Dict[str, str]] = None) -> str:
    """AWS_REGION → AWS_DEFAULT_REGION → botocore configured region (~/.aws/config profiles) → us-east-1."""
    env = env if env is not None else os.environ
    explicit = env.get("AWS_REGION", "").strip() or env.get("AWS_DEFAULT_REGION", "").strip()
    if explicit:
        return explicit
    with suppress(Exception):
        import botocore.session
        return botocore.session.get_session().get_config_variable("region") or "us-east-1"
    return "us-east-1"


def resolve_bedrock_runtime_region(config: Optional[Dict[str, Any]] = None) -> str:
    """``bedrock.region`` from config.yaml, else :func:`resolve_bedrock_region`. Every non-runtime Bedrock
    endpoint must use this so auxiliary calls never leave the primary runtime's region. *config* skips disk."""
    if config is None:
        with suppress(Exception):
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
    cfg_region = str(((config or {}).get("bedrock") or {}).get("region") or "").strip()
    return cfg_region or resolve_bedrock_region()


def bedrock_model_ids_or_none() -> Optional[List[str]]:
    """Live-discover Bedrock model IDs; None on failure/empty so callers use the static list."""
    with suppress(Exception):
        discovered = discover_bedrock_models(resolve_bedrock_runtime_region())
        if discovered:
            return merge_bedrock_openai_model_ids([m["id"] for m in discovered])
    return None


# --- Tool-calling / prompt-cache capability detection ---
# Models known to reject toolConfig with a ValidationException; unknown models assumed OK.
_NON_TOOL_CALLING_PATTERNS = [
    "deepseek.r1", "deepseek-r1",  # DeepSeek R1 (both ID formats) — reasoning only
    "stability.",  # image generation
    "cohere.embed", "amazon.titan-embed",  # embeddings
]

# cachePoint allowlist — inverted policy vs tools: unknown models get NO cache markers (they reject
# cachePoint). Claude only reaches build_converse_kwargs under bearer auth.
_CACHE_POINT_PATTERNS = ["anthropic.claude", "amazon.nova"]


def _model_supports_tool_use(model_id: str) -> bool:
    """False for denylisted models; unknown models default to True."""
    return not any(pattern in model_id.lower() for pattern in _NON_TOOL_CALLING_PATTERNS)


def _model_supports_prompt_cache(model_id: str) -> bool:
    return any(pattern in model_id.lower() for pattern in _CACHE_POINT_PATTERNS)


# --- Server-verdict cachePoint suppression ---
# Bedrock's cachePoint rule is per-family AND per-field (Nova accepts it in system/messages but hard-fails
# on toolConfig.tools) and any static table drifts, so when Bedrock names a placement as unpermitted we
# record the verdict, drop the marker there for the rest of the process, and retry once without it.

CACHE_POINT_PLACEMENTS = ("tools", "system", "messages")
_CACHE_POINT_REJECTIONS: Dict[str, set] = {}  # model_id (lowercased) → placements Bedrock has rejected this process
# e.g. "#/toolConfig/tools/18: extraneous key [cachePoint] is not permitted"
_CACHE_POINT_PATH_PATTERN = re.compile(r"#/(?P<path>[A-Za-z0-9_./\[\]-]*)", re.IGNORECASE)
_CACHE_POINT = {"cachePoint": {"type": "default"}}


def cache_point_rejection_placement(exc: BaseException) -> Optional[str]:
    """Converse section whose cachePoint Bedrock refused, or None. Message-based: the JSON pointer in the
    ValidationException is the only thing naming the section (raw or SDK-wrapped). Unlocalisable → "tools"."""
    msg = str(exc)
    lowered = msg.lower()
    if "cachepoint" not in lowered or ("not permitted" not in lowered and "extraneous" not in lowered):
        return None
    match = _CACHE_POINT_PATH_PATTERN.search(msg)
    path = (match.group("path") if match else "").lower()
    if "toolconfig" in path or "tools" in path:
        return "tools"
    return next((placement for placement in ("system", "messages") if placement in path), "tools")


def note_cache_point_rejection(model_id: str, placement: str) -> None:
    """Record that ``model_id`` refuses cachePoint blocks in ``placement``."""
    if placement in CACHE_POINT_PLACEMENTS:
        _CACHE_POINT_REJECTIONS.setdefault(model_id.lower(), set()).add(placement)


def cache_point_allowed(model_id: str, placement: str) -> bool:
    """False once Bedrock has refused this placement for this model."""
    return placement not in _CACHE_POINT_REJECTIONS.get(model_id.lower(), ())


def reset_cache_point_rejections() -> None:
    """Clear recorded cachePoint rejections. Used in tests."""
    _CACHE_POINT_REJECTIONS.clear()


def _without_cache_points(blocks: Any) -> Optional[list]:
    """``blocks`` minus cachePoint entries, or None if not a list / nothing removed."""
    if not isinstance(blocks, list):
        return None
    cleaned = [b for b in blocks if not (isinstance(b, dict) and set(b.keys()) == {"cachePoint"})]
    return None if len(cleaned) == len(blocks) else cleaned


def strip_cache_points(kwargs: Dict[str, Any], placement: str) -> Dict[str, Any]:
    """Copy of Converse kwargs with ``placement``'s cachePoint removed; the SAME object
    back when nothing was stripped (callers use identity to decide a retry cannot help)."""
    if placement == "messages":
        messages = kwargs.get("messages")
        cleaned_contents = [
            _without_cache_points(msg.get("content") if isinstance(msg, dict) else None) for msg in messages
        ] if isinstance(messages, list) else []
        if all(content is None for content in cleaned_contents):
            return kwargs
        return {**kwargs, "messages": [
            msg if content is None else {**msg, "content": content} for msg, content in zip(messages, cleaned_contents)
        ]}
    if placement == "system":
        cleaned = _without_cache_points(kwargs.get("system"))
        return kwargs if cleaned is None else {**kwargs, "system": cleaned}
    if placement == "tools":
        tool_config = kwargs.get("toolConfig")
        cleaned = _without_cache_points((tool_config or {}).get("tools"))
        return kwargs if cleaned is None else {**kwargs, "toolConfig": {**tool_config, "tools": cleaned}}
    return kwargs


def recover_from_cache_point_rejection(exc: BaseException, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Record Bedrock's cachePoint verdict and return retry kwargs, or None when the error
    was not a cachePoint rejection / the marker was already absent (caller re-raises)."""
    placement = cache_point_rejection_placement(exc)
    if placement is None:
        return None
    retry_kwargs = strip_cache_points(kwargs, placement)
    if retry_kwargs is kwargs:
        return None
    model_id = str(kwargs.get("modelId", ""))
    note_cache_point_rejection(model_id, placement)
    logger.warning(
        "bedrock: %s rejected a cachePoint block in %s — dropping that cache marker for this model and "
        "retrying. Prompt caching stays active for the remaining sections.", model_id or "model", placement,
    )
    return retry_kwargs


# One optional regional/global inference-profile prefix, then the Claude model family.
_ANTHROPIC_BEDROCK_MODEL_RE = re.compile(
    r"^(?:(?:global|us|eu|apac|ap|au|jp|ca|sa|me|af)\.)?anthropic\.claude", re.IGNORECASE,
)


def is_anthropic_bedrock_model(model_id: str) -> bool:
    """True for Claude on Bedrock (``anthropic.claude-*``, any regional prefix): AnthropicBedrock SDK path."""
    return _ANTHROPIC_BEDROCK_MODEL_RE.match(model_id) is not None


# --- Message format conversion: OpenAI → Bedrock Converse ---

def convert_tools_to_converse(tools: List[Dict]) -> List[Dict]:
    """OpenAI ``{"function": {...}}`` tool defs → Converse ``{"toolSpec": {...}}``."""
    return [{"toolSpec": {
        "name": fn.get("name", ""), "description": fn.get("description", ""),
        "inputSchema": {"json": fn.get("parameters", {"type": "object", "properties": {}})},
    }} for fn in (t.get("function", {}) for t in tools or [])]


# Converse rejects empty OR whitespace-only text blocks, so the placeholder must be non-whitespace.
# A lone space is whitespace and is rejected too — the placeholder MUST itself be non-whitespace. Ref: issue
# #9486.
_EMPTY_TEXT_PLACEHOLDER = "(empty)"
_PLACEHOLDER_BLOCK = {"text": _EMPTY_TEXT_PLACEHOLDER}


def _safe_text(text) -> str:
    """``text`` if it has non-whitespace content, else the placeholder (None/non-str ok)."""
    text = "" if text is None else str(text)
    return text if text.strip() else _EMPTY_TEXT_PLACEHOLDER


def _image_block_from_data_url(url: str) -> Dict:
    """``data:<mime>;base64,...`` → Converse image block with RAW bytes (boto3 base64-encodes on the
    wire; passing the string through double-encodes and Bedrock rejects it)."""
    header, _, data = url.partition(",")
    media_type = (header[5:].split(";")[0] if header.startswith("data:") else "") or "image/jpeg"
    try:
        # Ref: #33317.
        raw_bytes = base64.b64decode(data)
    except Exception:
        raw_bytes = data.encode("utf-8")
    return {"image": {"format": media_type.split("/")[-1] if "/" in media_type else "jpeg", "source": {"bytes": raw_bytes}}}


def _convert_content_to_converse(content) -> List[Dict]:
    """OpenAI content → Converse blocks; blank text → placeholder, remote image URLs → text reference."""
    if not isinstance(content, list):
        return [{"text": _safe_text(content)}]
    blocks = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"text": _safe_text(part)})
        elif isinstance(part, dict) and part.get("type", "") == "text":
            blocks.append({"text": _safe_text(part.get("text", ""))})
        elif isinstance(part, dict) and part.get("type", "") == "image_url":
            image_url = part.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
            blocks.append(_image_block_from_data_url(url) if url.startswith("data:") else {"text": f"[Image: {url}]"})
    return blocks or [dict(_PLACEHOLDER_BLOCK)]


def _system_blocks(content) -> List[Dict]:
    """System content → text blocks; blank parts are dropped, not placeholder-filled."""
    parts = [content] if isinstance(content, str) else content if isinstance(content, list) else []
    texts = [part.get("text", "") if isinstance(part, dict) and part.get("type") == "text" else part for part in parts]
    return [{"text": text} for text in texts if isinstance(text, str) and text.strip()]


def _tool_use_block(tool_use_id, name, input_dict) -> Dict:
    return {"toolUse": {"toolUseId": tool_use_id, "name": name, "input": input_dict}}


def _decode_redacted(encoded) -> Optional[bytes]:
    """Strict base64 → bytes; None for empty/non-str/undecodable input."""
    try:
        return base64.b64decode(encoded, validate=True) if isinstance(encoded, str) and encoded else None
    except (ValueError, TypeError):
        return None


def _replay_ordered_blocks(ordered_blocks: List) -> List[Dict]:
    """Rebuild the exact Bedrock block sequence captured at normalization time; redacted reasoning is
    stored base64 (JSON-safe sidecar) and undecodable entries are skipped."""
    content_blocks: List[Dict] = []
    for block in ordered_blocks:
        if not isinstance(block, dict):
            continue
        if "text" in block and isinstance(block["text"], str):
            content_blocks.append({"text": block["text"]})
        elif "reasoningContent" in block:
            reasoning = block["reasoningContent"]
            if not isinstance(reasoning, dict):
                continue
            replay = {"text": reasoning["text"]} if isinstance(reasoning.get("text"), str) else {}
            encoded = reasoning.get("redactedContentBase64")
            if isinstance(encoded, str) and encoded:
                redacted = _decode_redacted(encoded)
                if redacted is None:
                    continue
                replay["redactedContent"] = redacted
            if replay:
                content_blocks.append({"reasoningContent": replay})
        elif "toolUse" in block and isinstance(block["toolUse"], dict):
            tu = block["toolUse"]
            content_blocks.append(_tool_use_block(tu.get("toolUseId", ""), tu.get("name", ""), tu.get("input", {})))
    return content_blocks


def _parse_tool_args(args) -> Any:
    """JSON-decode a tool-call argument string; {} on failure; non-str passes through."""
    try:
        return json.loads(args) if isinstance(args, str) else args
    except (json.JSONDecodeError, TypeError):
        return {}


def _assistant_blocks(msg: Dict, content) -> List[Dict]:
    """Assistant message → Converse blocks. An ordered ``bedrock_content_blocks`` sidecar is authoritative;
    otherwise redacted thinking from ``reasoning_details`` (byte-for-byte), then text, then tool calls."""
    ordered_blocks = msg.get("bedrock_content_blocks")
    if isinstance(ordered_blocks, list) and (content_blocks := _replay_ordered_blocks(ordered_blocks)):
        return content_blocks
    redacted = [
        _decode_redacted(d.get("data") or d.get("redactedContentBase64"))
        for d in (msg.get("reasoning_details") or []) if isinstance(d, dict) and d.get("type") == "redacted_thinking"
    ]
    content_blocks: List[Dict] = [{"reasoningContent": {"redactedContent": r}} for r in redacted if r is not None]
    if isinstance(content, str) and content.strip():
        content_blocks.append({"text": content})
    elif isinstance(content, list):
        content_blocks.extend(_convert_content_to_converse(content))
    for tc in (msg.get("tool_calls", []) or []):
        fn = tc.get("function", {})
        content_blocks.append(_tool_use_block(tc.get("id", ""), fn.get("name", ""), _parse_tool_args(fn.get("arguments", "{}"))))
    return content_blocks


def convert_messages_to_converse(messages: List[Dict]) -> Tuple[Optional[List[Dict]], List[Dict]]:
    """OpenAI messages → ``(system_blocks_or_None, converse_messages)``; tool results become ``toolResult``
    user blocks. Converse needs strict user/assistant alternation with a user turn first and last:
    same-role neighbours merge, placeholder user turns pad the ends."""
    system_blocks: List[Dict] = []
    converse_msgs: List[Dict] = []

    def append_turn(role: str, blocks: List[Dict]) -> None:
        if converse_msgs and converse_msgs[-1]["role"] == role:
            converse_msgs[-1]["content"].extend(blocks)
        else:
            converse_msgs.append({"role": role, "content": blocks})

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")
        if role == "system":
            system_blocks.extend(_system_blocks(content))
        elif role == "tool":
            result_content = content if isinstance(content, str) else json.dumps(content)
            append_turn("user", [{"toolResult": {
                "toolUseId": msg.get("tool_call_id", ""), "content": [{"text": _safe_text(result_content)}]}}])
        elif role == "assistant":
            append_turn("assistant", _assistant_blocks(msg, content) or [dict(_PLACEHOLDER_BLOCK)])
        elif role == "user":
            append_turn("user", _convert_content_to_converse(content))
    if converse_msgs and converse_msgs[0]["role"] != "user":
        converse_msgs.insert(0, {"role": "user", "content": [dict(_PLACEHOLDER_BLOCK)]})
    if converse_msgs and converse_msgs[-1]["role"] != "user":
        converse_msgs.append({"role": "user", "content": [dict(_PLACEHOLDER_BLOCK)]})
    return (system_blocks or None, converse_msgs)


# --- Response format conversion: Bedrock Converse → OpenAI ---

# Bedrock stopReason → OpenAI finish_reason (unknown → "stop").
_STOP_REASON_TO_FINISH_REASON = {
    "end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length",
    "content_filtered": "content_filter", "guardrail_intervened": "content_filter",
}


def _encode_redacted(redacted) -> Optional[str]:
    """Redacted reasoning payload → base64 str (bytes encoded, str passed through, else None)."""
    if isinstance(redacted, (bytes, bytearray)):
        return base64.b64encode(bytes(redacted)).decode("ascii")
    return redacted if isinstance(redacted, str) else None


def _tool_call_ns(tool_use_id: str, name: str, input_dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=tool_use_id, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(input_dict)),
    )


class _ResponseParts:
    """Accumulator shared by the sync and streaming normalizers."""

    def __init__(self) -> None:
        self.text_parts: List[str] = []
        self.reasoning_parts: List[str] = []
        self.reasoning_details: List[Dict[str, Any]] = []
        self.tool_calls: List[SimpleNamespace] = []

    def absorb_reasoning(self, reasoning: Any, block: Dict[str, Any], on_text=None) -> None:
        """Fold a Converse ``reasoningContent`` payload into the accumulators and ``block``."""
        if not isinstance(reasoning, dict):
            return
        thinking_text = reasoning.get("text", "")
        if thinking_text:
            self.reasoning_parts.append(str(thinking_text))
            if on_text:
                on_text(thinking_text)
            block["text"] = block.get("text", "") + str(thinking_text)
        encoded = _encode_redacted(reasoning.get("redactedContent"))
        if encoded:
            self.reasoning_details.append({"type": "redacted_thinking", "data": encoded})
            block["redactedContentBase64"] = encoded

    def build(self, ordered_blocks: List[Dict[str, Any]], usage_data: Dict[str, int], stop_reason: str, model: str) -> SimpleNamespace:
        """Assemble the OpenAI-shaped response. Converse's inputTokens EXCLUDES cache read/write tokens
        (OpenAI's prompt_tokens includes them), so they are added back."""
        msg = SimpleNamespace(
            role="assistant", content="\n".join(self.text_parts) if self.text_parts else None,
            tool_calls=self.tool_calls or None, reasoning_details=self.reasoning_details or None,
            reasoning_content="\n\n".join(self.reasoning_parts) if self.reasoning_parts else None,
            bedrock_content_blocks=ordered_blocks or None,
        )
        cache_read_tokens, cache_write_tokens, output_tokens = (
            usage_data.get(k, 0) for k in ("cacheReadInputTokens", "cacheWriteInputTokens", "outputTokens")
        )
        prompt_tokens = usage_data.get("inputTokens", 0) + cache_read_tokens + cache_write_tokens
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=output_tokens, total_tokens=prompt_tokens + output_tokens,
            cache_read_input_tokens=cache_read_tokens, cache_creation_input_tokens=cache_write_tokens,
        )
        finish_reason = _STOP_REASON_TO_FINISH_REASON.get(stop_reason, "stop")
        if self.tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        return SimpleNamespace(
            choices=[SimpleNamespace(index=0, message=msg, finish_reason=finish_reason)], usage=usage, model=model,
        )


def normalize_converse_response(response: Dict) -> SimpleNamespace:
    """Bedrock Converse response → OpenAI ``ChatCompletion``-shaped SimpleNamespace (``.choices[0].message.
    {content,tool_calls,reasoning_content,reasoning_details,bedrock_content_blocks}``, ``finish_reason``, ``.usage``)."""
    parts = _ResponseParts()
    ordered_blocks = []
    for block in response.get("output", {}).get("message", {}).get("content", []):
        if "text" in block:
            parts.text_parts.append(block["text"])
            ordered_blocks.append({"text": block["text"]})
        elif "reasoningContent" in block:
            ordered_reasoning: Dict[str, Any] = {}
            parts.absorb_reasoning(block["reasoningContent"], ordered_reasoning)
            if ordered_reasoning:
                ordered_blocks.append({"reasoningContent": ordered_reasoning})
        elif "toolUse" in block:
            tu = block["toolUse"]
            ordered_blocks.append(_tool_use_block(tu.get("toolUseId", ""), tu.get("name", ""), tu.get("input", {})))
            parts.tool_calls.append(_tool_call_ns(tu.get("toolUseId", ""), tu.get("name", ""), tu.get("input", {})))
    return parts.build(
        ordered_blocks, response.get("usage", {}), response.get("stopReason", "end_turn"), response.get("modelId", ""),
    )


# --- Streaming response conversion ---

def normalize_converse_stream_events(event_stream) -> SimpleNamespace:
    """Consume a ConverseStream event stream (no callbacks) → same shape as ``normalize_converse_response()``."""
    return stream_converse_with_callbacks(event_stream)


def stream_converse_with_callbacks(
    event_stream, on_text_delta=None, on_tool_start=None, on_reasoning_delta=None,
    on_interrupt_check=None, on_event=None,
) -> SimpleNamespace:
    """boto3 ``converse_stream()`` response + callbacks → the ``normalize_converse_response()`` shape.
    ``on_text_delta`` only fires while no toolUse block has been seen (as on the Anthropic/chat_completions
    paths); ``on_interrupt_check`` True stops streaming; ``on_event`` fires for EVERY event before branching
    and its exceptions are swallowed so a watchdog hook can never abort the stream."""
    parts = _ResponseParts()
    stream_blocks: Dict[int, Dict[str, Any]] = {}
    current_block_index: Optional[int] = None
    current_tool: Optional[Dict] = None
    current_text_buffer: List[str] = []
    has_tool_use = False
    stop_reason = "end_turn"
    usage_data: Dict[str, int] = {}

    def current_block(default: Dict[str, Any]) -> Dict[str, Any]:
        idx = current_block_index if current_block_index is not None else len(stream_blocks)
        return stream_blocks.setdefault(idx, default)

    def flush_text() -> None:
        if current_text_buffer:
            parts.text_parts.append("".join(current_text_buffer))
            current_text_buffer.clear()

    for event in event_stream.get("stream", []):
        if on_event is not None:
            with suppress(Exception):
                on_event()
        if on_interrupt_check and on_interrupt_check():
            break
        if "contentBlockStart" in event:
            start_event = event["contentBlockStart"]
            current_block_index = start_event.get("contentBlockIndex", len(stream_blocks))
            start = start_event.get("start", {})
            if "toolUse" in start:
                has_tool_use = True
                flush_text()
                current_tool = {"toolUseId": start["toolUse"].get("toolUseId", ""), "name": start["toolUse"].get("name", ""), "input_json": ""}
                stream_blocks[current_block_index] = _tool_use_block(current_tool["toolUseId"], current_tool["name"], {})
                if on_tool_start:
                    on_tool_start(current_tool["name"])
        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                text = delta["text"]
                block = current_block({"text": ""})
                block["text"] = block.get("text", "") + text
                current_text_buffer.append(text)
                if on_text_delta and not has_tool_use:
                    on_text_delta(text)
            elif "toolUse" in delta and current_tool is not None:
                current_tool["input_json"] += delta["toolUse"].get("input", "")
            elif "reasoningContent" in delta:
                reasoning = delta["reasoningContent"]
                if isinstance(reasoning, dict) and (reasoning.get("text", "") or _encode_redacted(reasoning.get("redactedContent"))):
                    block = current_block({"reasoningContent": {}}).setdefault("reasoningContent", {})
                    parts.absorb_reasoning(reasoning, block, on_reasoning_delta)
        elif "contentBlockStop" in event:
            if current_tool is not None:
                input_dict = _parse_tool_args(current_tool["input_json"])  # "" → {} via the JSON-error path
                parts.tool_calls.append(_tool_call_ns(current_tool["toolUseId"], current_tool["name"], input_dict))
                if current_block_index is not None and current_block_index in stream_blocks:
                    stream_blocks[current_block_index]["toolUse"]["input"] = input_dict
                current_tool = None
            else:
                flush_text()
        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "end_turn")
        elif "metadata" in event:
            meta_usage = event["metadata"].get("usage", {})
            usage_data = {key: meta_usage.get(key, 0) for key in ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheWriteInputTokens")}
    flush_text()
    return parts.build([stream_blocks[i] for i in sorted(stream_blocks)], usage_data, stop_reason, "")


# --- High-level API: call Bedrock Converse ---

def build_converse_kwargs(
    model: str, messages: List[Dict], tools: Optional[List[Dict]] = None, max_tokens: Optional[int] = 4096,
    temperature: Optional[float] = None, top_p: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None, guardrail_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build kwargs for ``bedrock-runtime.converse()`` / ``converse_stream()``. ``max_tokens=None`` omits
    ``maxTokens`` (model maximum; default stays 4096). cachePoint markers go on system, tools and the
    second-newest message (survives as the tail grows — mirrors Anthropic system_and_3), each only if the
    model supports caching and Bedrock has not rejected that placement."""
    system_prompt, converse_messages = convert_messages_to_converse(messages)
    cache_at = {p for p in CACHE_POINT_PLACEMENTS if cache_point_allowed(model, p)} if _model_supports_prompt_cache(model) else set()
    inference_config: Dict[str, Any] = {} if max_tokens is None else {"maxTokens": max_tokens}
    kwargs: Dict[str, Any] = {"modelId": model, "messages": converse_messages, "inferenceConfig": inference_config}
    if system_prompt:
        kwargs["system"] = system_prompt + [dict(_CACHE_POINT)] if "system" in cache_at else system_prompt
    from agent.anthropic_adapter import _forbids_sampling_params
    if not _forbids_sampling_params(model):
        inference_config.update({k: v for k, v in (("temperature", temperature), ("topP", top_p)) if v is not None})
    if stop_sequences:
        inference_config["stopSequences"] = stop_sequences
    converse_tools = convert_tools_to_converse(tools) if tools else []
    if converse_tools and not _model_supports_tool_use(model):
        # Non-tool-calling models reject toolConfig (ValidationException → retry loop): strip and warn.
        logger.warning(
            "Model %s does not support tool calling — tools stripped. "
            "The agent will operate in text-only mode.", model
        )
    elif converse_tools:
        kwargs["toolConfig"] = {"tools": converse_tools + [dict(_CACHE_POINT)] if "tools" in cache_at else converse_tools}
    if "messages" in cache_at and len(converse_messages) >= 2:
        content = converse_messages[-2].get("content")
        if isinstance(content, list) and content:
            content.append(dict(_CACHE_POINT))
    if guardrail_config:
        kwargs["guardrailConfig"] = guardrail_config
    if not inference_config:
        del kwargs["inferenceConfig"]  # optional on the wire; don't send {}
    return kwargs


def call_converse(
    region: str, model: str, messages: List[Dict], tools: Optional[List[Dict]] = None,
    max_tokens: Optional[int] = 4096, temperature: Optional[float] = None, top_p: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None, guardrail_config: Optional[Dict] = None,
) -> SimpleNamespace:
    """Non-streaming Converse call → OpenAI-compatible response. Retries once without a rejected cachePoint
    placement; evicts the cached client on stale-connection errors."""
    client = _get_bedrock_runtime_client(region)
    kwargs = build_converse_kwargs(model, messages, tools, max_tokens, temperature, top_p, stop_sequences, guardrail_config)
    try:
        response = client.converse(**kwargs)
    except Exception as exc:
        retry_kwargs = recover_from_cache_point_rejection(exc, kwargs)
        if retry_kwargs is not None:
            return normalize_converse_response(client.converse(**retry_kwargs))
        if is_stale_connection_error(exc):
            logger.warning(
                "bedrock: stale-connection error on converse(region=%s, model=%s): "
                "%s — evicting cached client so the next call reconnects.", region, model, type(exc).__name__,
            )
            invalidate_runtime_client(region)
        raise
    return normalize_converse_response(response)


# --- Model discovery ---

_discovery_cache: Dict[str, Any] = {}
_DISCOVERY_CACHE_TTL_SECONDS = 3600


def reset_discovery_cache():
    """Clear the model discovery cache. Used in tests."""
    _discovery_cache.clear()


def _model_entry(model_id: str, name: Any, provider: str, input_mods: list, output_mods: list) -> Dict[str, Any]:
    return {"id": model_id, "name": (name or model_id).strip(), "provider": provider,
            "input_modalities": input_mods, "output_modalities": output_mods, "streaming": True}


def _list_foundation_models(client, filter_set: set, models: List[Dict[str, Any]]) -> None:
    """Append active, streaming-capable, text-output foundation models (optionally provider-filtered)."""
    for summary in client.list_foundation_models().get("modelSummaries", []):
        model_id = (summary.get("modelId") or "").strip()
        if not model_id:
            continue
        provider_name = summary.get("providerName") or ""
        model_prefix = model_id.split(".")[0].lower() if "." in model_id else ""
        if filter_set and provider_name.lower() not in filter_set and model_prefix not in filter_set:
            continue
        output_mods = summary.get("outputModalities", [])
        if (summary.get("modelLifecycle", {}).get("status", "").upper() != "ACTIVE"
                or not summary.get("responseStreamingSupported", False) or "TEXT" not in output_mods):
            continue
        models.append(_model_entry(
            model_id, summary.get("modelName"), provider_name.strip(), summary.get("inputModalities", []), output_mods,
        ))


def _list_inference_profiles(client, filter_set: set, models: List[Dict[str, Any]]) -> None:
    """Append active cross-region inference profiles whose IDs are not already present (paginated)."""
    profiles, next_token = [], None
    while True:
        response = client.list_inference_profiles(**({"nextToken": next_token} if next_token else {}))
        profiles.extend(response.get("inferenceProfileSummaries", []))
        if not (next_token := response.get("nextToken")):
            break
    seen_ids = {m["id"].lower() for m in models}
    for profile in profiles:
        profile_id = (profile.get("inferenceProfileId") or "").strip()
        if not profile_id or profile.get("status") != "ACTIVE" or profile_id.lower() in seen_ids:
            continue
        if filter_set and not any(
            _extract_provider_from_arn(m.get("modelArn", "")).lower() in filter_set for m in profile.get("models", [])
        ):
            continue
        models.append(_model_entry(profile_id, profile.get("inferenceProfileName"), "inference-profile", ["TEXT"], ["TEXT"]))
        seen_ids.add(profile_id.lower())


def discover_bedrock_models(region: str, provider_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Foundation models + inference profiles (cached 1h per region/filter), ``global.`` profiles first then
    by name; [] when the client cannot be built."""
    cache_key = f"{region}:{','.join(sorted(provider_filter or []))}"
    cached = _discovery_cache.get(cache_key)
    if cached and (time.time() - cached["timestamp"]) < _DISCOVERY_CACHE_TTL_SECONDS:
        return cached["models"]
    try:
        client = _get_bedrock_control_client(region)
    except Exception as e:
        logger.warning("Failed to create Bedrock client for model discovery: %s", e)
        return []
    models: List[Dict[str, Any]] = []
    filter_set = {f.lower() for f in (provider_filter or [])}
    for step, log, message in (
        (_list_foundation_models, logger.warning, "Failed to list Bedrock foundation models: %s"),
        (_list_inference_profiles, logger.debug, "Skipping inference profile discovery: %s"),
    ):
        try:
            step(client, filter_set, models)
        except Exception as e:
            log(message, e)
    models.sort(key=lambda m: (0 if m["id"].startswith("global.") else 1, m["name"].lower()))
    _discovery_cache[cache_key] = {"timestamp": time.time(), "models": models}
    return models


def _extract_provider_from_arn(arn: str) -> str:
    """``arn:aws:bedrock:...:foundation-model/anthropic.claude-v2`` → ``"anthropic"``."""
    match = re.search(r"foundation-model/([^.]+)", arn)
    return match.group(1) if match else ""


# --- Bedrock model context lengths ---
# Static fallback when the live probe is unavailable (agent/model_metadata.py). Keys match by longest
# substring, so versioned entries win over the generic "anthropic.claude-opus-4".

BEDROCK_CONTEXT_LENGTHS: Dict[str, int] = {
    # Anthropic Claude: 1M GA vs 200K. The 1M entries must match agent/model_metadata.py
    # DEFAULT_CONTEXT_LENGTHS or context compresses early.
    **dict.fromkeys((
        "anthropic.claude-fable-5", "anthropic.claude-fable", "anthropic.claude-sonnet-5", "anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-7", "anthropic.claude-opus-4-6", "anthropic.claude-sonnet-4-6",
    ), 1_000_000),
    **dict.fromkeys((
        "anthropic.claude-sonnet-4-5", "anthropic.claude-haiku-4-5", "anthropic.claude-opus-4", "anthropic.claude-sonnet-4",
        "anthropic.claude-3-5-sonnet", "anthropic.claude-3-5-haiku", "anthropic.claude-3-opus", "anthropic.claude-3-sonnet",
        "anthropic.claude-3-haiku",
    ), 200_000),
    # Amazon Nova
    **dict.fromkeys(("amazon.nova-pro", "amazon.nova-lite"), 300_000), "amazon.nova-micro": 128_000,
    # Meta Llama / Mistral / DeepSeek
    **dict.fromkeys((
        "meta.llama4-maverick", "meta.llama4-scout", "meta.llama3-3-70b-instruct", "mistral.mistral-large", "deepseek.v3",
    ), 128_000),
    # OpenAI on Bedrock (Mantle/Responses route): docs.aws.amazon.com/bedrock/latest/userguide/model-cards-openai.html
    **dict.fromkeys(BEDROCK_OPENAI_RESPONSES_MODEL_IDS, 272_000),
}

BEDROCK_DEFAULT_CONTEXT_LENGTH = 128_000  # unknown Bedrock models

# Probe padding tiers (tokens): a wildly oversized payload yields an opaque InternalServerException
# instead of a clean ValidationException.
_BEDROCK_PROBE_TIERS = (1_300_000, 2_200_000)
_WORDS_PER_TOKEN = 0.9  # conservative: ensures the padded prompt clears the tier


def probe_bedrock_context_length(model_id: str, region: str) -> Optional[int]:
    """Discover a model's real context window by provoking a length error — the only authoritative source
    ("prompt is too long: 1300032 tokens > 1000000 maximum"); length validation runs before inference so the
    probe costs nothing. An accepted tier is a safe lower bound; None (no creds/network/unparseable) → static table."""
    from agent.model_metadata import parse_context_limit_from_error
    try:
        client = _get_bedrock_runtime_client(region)
    except Exception as exc:  # boto3 missing / credential resolution failure
        logger.debug("Bedrock context probe skipped for %s: %s", model_id, exc)
        return None
    last_error = ""
    for tier_tokens in _BEDROCK_PROBE_TIERS:
        oversized = "data " * int(tier_tokens / _WORDS_PER_TOKEN)
        try:
            client.converse(modelId=model_id, messages=[{"role": "user", "content": [{"text": oversized}]}],
                            inferenceConfig={"maxTokens": 8})
            logger.debug("Bedrock context probe for %s accepted ~%s-token prompt; "
                         "window is at least that", model_id, f"{tier_tokens:,}")
            return tier_tokens
        except Exception as exc:
            last_error = str(exc)
            limit = parse_context_limit_from_error(last_error)
            if limit and limit >= 1024:
                logger.info("Probed Bedrock context window for %s: %s tokens", model_id, f"{limit:,}")
                return limit
            # Opaque server error / auth / throttle at this tier — try the next.
    logger.debug("Bedrock context probe for %s returned no parseable limit: %s", model_id, last_error[:200])
    return None


def get_bedrock_context_length(model_id: str, region: str = "", probe: bool = True) -> int:
    """Context window: live probe (if ``probe`` and ``region``) → static table → default. The table is fallback
    only: a stale substring match silently caps the window (a 1M Opus pinned to 200K via "opus-4")."""
    if probe and region and (probed := probe_bedrock_context_length(model_id, region)):
        return probed
    matches = [key for key in BEDROCK_CONTEXT_LENGTHS if key in model_id.lower()]
    return BEDROCK_CONTEXT_LENGTHS[max(matches, key=len)] if matches else BEDROCK_DEFAULT_CONTEXT_LENGTH


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

CONTEXT_OVERFLOW_PATTERNS = [
    re.compile(r"ValidationException.*(?:input is too long|max input token|input token.*exceed)", re.IGNORECASE),
    re.compile(r"ValidationException.*(?:exceeds? the (?:maximum|max) (?:number of )?(?:input )?tokens)", re.IGNORECASE),
    re.compile(r"ModelStreamErrorException.*(?:Input is too long|too many input tokens)", re.IGNORECASE),
]

OVERLOAD_PATTERNS = [
    re.compile(r"ModelNotReadyException", re.IGNORECASE),
    re.compile(r"ModelTimeoutException", re.IGNORECASE),
    re.compile(r"InternalServerException", re.IGNORECASE),
]

THROTTLE_PATTERNS = [
    re.compile(r"ThrottlingException", re.IGNORECASE),
    re.compile(r"Too many concurrent requests", re.IGNORECASE),
    re.compile(r"ServiceQuotaExceededException", re.IGNORECASE),
]

def call_converse_stream(
    region: str,
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    max_tokens: Optional[int] = 4096,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
    guardrail_config: Optional[Dict] = None,
) -> SimpleNamespace:
    """Call Bedrock ConverseStream API and return an OpenAI-compatible response.

    Consumes the full stream and returns the assembled response. For true
    streaming with delta callbacks, use ``iter_converse_stream()`` instead.
    """
    client = _get_bedrock_runtime_client(region)
    kwargs = build_converse_kwargs(
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=stop_sequences,
        guardrail_config=guardrail_config,
    )

    try:
        response = client.converse_stream(**kwargs)
    except Exception as exc:
        retry_kwargs = recover_from_cache_point_rejection(exc, kwargs)
        if retry_kwargs is not None:
            return normalize_converse_stream_events(
                client.converse_stream(**retry_kwargs)
            )
        if is_streaming_access_denied_error(exc):
            # IAM allows bedrock:InvokeModel but not
            # InvokeModelWithResponseStream — permanent for this session.
            # Fall back to the non-streaming converse() path.
            logger.info(
                "bedrock: converse_stream denied by IAM on (region=%s, model=%s) — "
                "falling back to non-streaming converse().",
                region, model,
            )
            return normalize_converse_response(client.converse(**kwargs))
        if is_stale_connection_error(exc):
            logger.warning(
                "bedrock: stale-connection error on converse_stream(region=%s, "
                "model=%s): %s — evicting cached client so the next call reconnects.",
                region, model, type(exc).__name__,
            )
            invalidate_runtime_client(region)
        raise
    return normalize_converse_stream_events(response)

def is_context_overflow_error(error_message: str) -> bool:
    """Return True if the error indicates the input context was too large.

    When this returns True, the agent should compress context and retry
    rather than treating it as a fatal error.
    """
    return any(p.search(error_message) for p in CONTEXT_OVERFLOW_PATTERNS)

def classify_bedrock_error(error_message: str) -> str:
    """Classify a Bedrock error for retry/failover decisions.

    Returns:
      - ``"context_overflow"`` — input too long, compress and retry
      - ``"rate_limit"`` — throttled, backoff and retry
      - ``"overloaded"`` — model temporarily unavailable, retry with delay
      - ``"unknown"`` — unclassified error
    """
    if is_context_overflow_error(error_message):
        return "context_overflow"
    if any(p.search(error_message) for p in THROTTLE_PATTERNS):
        return "rate_limit"
    if any(p.search(error_message) for p in OVERLOAD_PATTERNS):
        return "overloaded"
    return "unknown"
# ---- END PLUGIN-COMPAT ----
