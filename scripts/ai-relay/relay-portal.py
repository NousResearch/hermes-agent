#!/usr/bin/env python3
"""relay-portal — call AI Portal routes without requiring local vendor login."""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


TOKEN_ENVS = {
    "claude": ("AI_PORTAL_CLAUDE_TOKEN", "AI_RELAY_CLAUDE_TOKEN", "AI_PORTAL_TOKEN"),
    "opus": ("AI_PORTAL_CLAUDE_TOKEN", "AI_RELAY_CLAUDE_TOKEN", "AI_PORTAL_TOKEN"),
    "codex": ("AI_PORTAL_CODEX_TOKEN", "AI_RELAY_CODEX_TOKEN", "OPENAI_API_KEY"),
    "openai": ("AI_PORTAL_CODEX_TOKEN", "AI_RELAY_CODEX_TOKEN", "OPENAI_API_KEY"),
    "grok": ("AI_PORTAL_GROK_TOKEN", "AI_RELAY_GROK_TOKEN", "GROK_API_KEY"),
}

CODEX_DEFAULT_IDENTITY = {
    "01": {
        "email": "jigsawaiteam@gmail.com",
        "user_id": "user-Jcm2OkAJD6NGd8wB5KtDUHLN",
    },
    "02": {
        "email": "jigsawgroupscompany@gmail.com",
        "user_id": "user-xIuPjll2xHbeeMTCMYY84TiA",
    },
}

DEFAULT_MODELS = {
    "claude": "claude-opus-4-8",
    "opus": "claude-opus-4-8",
    "codex": "gpt-5.5",
    "openai": "gpt-5.5",
    "grok": "grok-4.5",
}


def portal_base() -> str:
    raw = (
        os.environ.get("AI_PORTAL_BASE_URL")
        or os.environ.get("AI_PORTAL_URL")
        or "http://103.142.150.185:3012"
    ).rstrip("/")
    return raw[:-3] if raw.endswith("/v1") else raw


def token_for(provider: str) -> str:
    if provider in ("codex", "openai"):
        routed = codex_cross_route_token()
        if routed:
            return routed
    for name in TOKEN_ENVS.get(provider, ()):
        value = os.environ.get(name)
        if value:
            return value
    names = ", ".join(TOKEN_ENVS.get(provider, ()))
    raise SystemExit(f"auth: missing portal token env for {provider} ({names})")


def decode_jwt_payload(token: str | None) -> dict:
    if not token or token.count(".") < 2:
        return {}
    payload = token.split(".")[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload.encode("utf-8")))
    except Exception:
        return {}


def codex_local_identity() -> dict:
    forced_email = os.environ.get("AI_PORTAL_CODEX_LOGIN_EMAIL")
    forced_user = os.environ.get("AI_PORTAL_CODEX_LOGIN_USER_ID")
    if forced_email or forced_user:
        return {"email": forced_email, "user_id": forced_user, "source": "env"}

    path = Path(os.environ.get("CODEX_AUTH_FILE", str(Path.home() / ".codex" / "auth.json")))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"source": str(path)}

    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
    id_payload = decode_jwt_payload(tokens.get("id_token") or data.get("id_token"))
    access_payload = decode_jwt_payload(tokens.get("access_token") or data.get("access_token"))
    profile = access_payload.get("https://api.openai.com/profile") or {}
    auth = (
        id_payload.get("https://api.openai.com/auth")
        or access_payload.get("https://api.openai.com/auth")
        or {}
    )
    return {
        "email": data.get("email") or id_payload.get("email") or profile.get("email"),
        "user_id": data.get("user_id") or auth.get("user_id"),
        "source": str(path),
    }


def codex_identity_matches(identity: dict, slot: str) -> bool:
    email = (identity.get("email") or "").strip().lower()
    user_id = (identity.get("user_id") or "").strip()
    expected_email = os.environ.get(f"AI_PORTAL_CODEX_{slot}_EMAIL") or CODEX_DEFAULT_IDENTITY[slot]["email"]
    expected_user = os.environ.get(f"AI_PORTAL_CODEX_{slot}_USER_ID") or CODEX_DEFAULT_IDENTITY[slot]["user_id"]
    return bool(
        (email and email == expected_email.strip().lower())
        or (user_id and user_id == expected_user.strip())
    )


def codex_cross_route_token() -> str | None:
    token_01 = os.environ.get("AI_PORTAL_CODEX_TOKEN_01")
    token_02 = os.environ.get("AI_PORTAL_CODEX_TOKEN_02")
    if not token_01 or not token_02:
        return None

    identity = codex_local_identity()
    if codex_identity_matches(identity, "01"):
        return token_02
    if codex_identity_matches(identity, "02"):
        return token_01

    target = os.environ.get("AI_PORTAL_CODEX_DEFAULT_CROSS_TARGET", "").strip()
    if target == "01":
        return token_01
    if target == "02":
        return token_02
    if os.environ.get("AI_PORTAL_CODEX_CROSS_ROUTE", "strict").lower() in ("0", "false", "off"):
        return os.environ.get("AI_PORTAL_CODEX_TOKEN")

    seen = ", ".join(
        f"{key}={value}"
        for key, value in [("email", identity.get("email")), ("user_id", identity.get("user_id")), ("source", identity.get("source"))]
        if value
    )
    raise SystemExit(
        "auth: cannot detect local Codex login for cross-route. "
        "Set AI_PORTAL_CODEX_LOGIN_EMAIL or AI_PORTAL_CODEX_DEFAULT_CROSS_TARGET=01|02. "
        f"Detected: {seen or 'none'}"
    )


def request_json(url: str, token: str, payload: dict) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
            "user-agent": "ai-relay-portal/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=int(os.environ.get("AI_PORTAL_TIMEOUT", "300"))) as resp:
            return resp.read()
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", "replace")
        if err.code in (401, 403):
            raise SystemExit(f"auth: AI Portal rejected token ({err.code}) {body}")
        if err.code == 429:
            raise SystemExit(f"quota: AI Portal rate limited ({err.code}) {body}")
        raise SystemExit(f"crash: AI Portal error ({err.code}) {body}")
    except urllib.error.URLError as err:
        raise SystemExit(f"crash: cannot reach AI Portal: {err.reason}")


def sse_text(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace")
    out = []
    for block in text.split("\n\n"):
        data_lines = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        if not data or data == "[DONE]":
            continue
        try:
            value = json.loads(data)
        except json.JSONDecodeError:
            continue
        event_type = value.get("type", "")
        if event_type == "response.output_text.delta":
            out.append(value.get("delta", ""))
    return "".join(out).strip()


def response_text(raw: bytes, provider: str) -> str:
    if raw.lstrip().startswith(b"event:"):
        return sse_text(raw)
    body = json.loads(raw.decode("utf-8", "replace"))
    if provider in ("claude", "opus"):
        parts = []
        for item in body.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    if provider in ("grok",):
        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return (message.get("content") or choices[0].get("text") or "").strip()
    if provider in ("codex", "openai"):
        if isinstance(body.get("output_text"), str):
            return body["output_text"].strip()
    return json.dumps(body, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=sorted(TOKEN_ENVS))
    parser.add_argument("--model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("AI_PORTAL_MAX_TOKENS", "4096")))
    args = parser.parse_args()

    provider = args.provider
    model = args.model or DEFAULT_MODELS[provider]
    base = portal_base()
    token = token_for(provider)

    if provider in ("claude", "opus"):
        url = f"{base}/v1/messages"
        payload = {
            "model": model,
            "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": args.prompt}],
        }
    elif provider in ("codex", "openai"):
        url = f"{base}/v1/responses"
        payload = {"model": model, "input": args.prompt, "max_output_tokens": args.max_tokens}
    else:
        url = f"{base}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
        }

    raw = request_json(url, token, payload)
    print(response_text(raw, provider))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
