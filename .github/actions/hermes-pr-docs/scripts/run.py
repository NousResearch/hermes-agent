#!/usr/bin/env python3
import json
import os
import sys
import subprocess
import time
import urllib.request

API = "https://api.github.com"


def gh_api(url, token):
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def gh_api_paginate(url, token, limit=200):
    results = []
    page = 1
    while True:
        u = f"{url}?per_page=100&page={page}"
        data = gh_api(u, token)
        if not data:
            break
        results.extend(data)
        if len(data) < 100 or len(results) >= limit:
            break
        page += 1
    return results[:limit]


def get_env(name, default=None, required=False):
    val = os.getenv(name, default)
    if required and not val:
        print(f"Missing env: {name}", file=sys.stderr)
        sys.exit(2)
    return val


def emit_github_output(name, value):
    value = value or ""
    target = os.getenv("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(f"{name}<<EOF\n{value}\nEOF\n")
        return
    print(f"{name}<<EOF\n{value}\nEOF")


def get_github_app_token(app_id=None, installation_id=None, private_key=None, private_key_path=None):
    app_id = app_id or os.getenv("GITHUB_APP_ID") or os.getenv("INPUT_GITHUB_APP_ID")
    installation_id = installation_id or os.getenv("GITHUB_APP_INSTALLATION_ID") or os.getenv("INPUT_GITHUB_APP_INSTALLATION_ID")
    private_key = private_key or os.getenv("GITHUB_APP_PRIVATE_KEY") or os.getenv("INPUT_GITHUB_APP_PRIVATE_KEY")
    private_key_path = private_key_path or os.getenv("GITHUB_APP_PRIVATE_KEY_PATH") or os.getenv("INPUT_GITHUB_APP_PRIVATE_KEY_PATH")

    if not app_id or not installation_id:
        raise ValueError("GitHub App requires GITHUB_APP_ID and GITHUB_APP_INSTALLATION_ID")

    if not private_key and private_key_path:
        with open(private_key_path, "r", encoding="utf-8") as fh:
            private_key = fh.read()

    if not private_key:
        raise ValueError("GitHub App requires GITHUB_APP_PRIVATE_KEY or GITHUB_APP_PRIVATE_KEY_PATH")

    try:
        import jwt
    except ImportError as exc:
        raise RuntimeError("PyJWT is required for GitHub App authentication") from exc

    clean_key = private_key.replace("\\n", "\n").strip()
    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + (10 * 60),
        "iss": str(app_id),
    }
    jwt_token = jwt.encode(payload, clean_key, algorithm="RS256")

    req = urllib.request.Request(
        f"{API}/app/installations/{installation_id}/access_tokens",
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {jwt_token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.load(r)

    token = body.get("token")
    if not token:
        raise ValueError("GitHub App access token response missing token")
    return token


def resolve_github_token():
    for env_name in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.getenv(env_name, "").strip()
        if token:
            return token

    try:
        return get_github_app_token()
    except Exception as exc:
        print(f"GitHub App token fallback failed: {exc}", file=sys.stderr)
        return ""


def post_pr_review(token, repo, pr_number, body, review_mode="comment"):
    review_mode = (review_mode or "comment").lower()
    if review_mode == "review":
        endpoint = f"{API}/repos/{repo}/pulls/{pr_number}/reviews"
        payload = json.dumps({
            "event": "COMMENT",
            "body": body,
        }).encode("utf-8")
    else:
        endpoint = f"{API}/repos/{repo}/issues/{pr_number}/comments"
        payload = json.dumps({"body": body}).encode("utf-8")

    req = urllib.request.Request(endpoint, data=payload, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    token = resolve_github_token()
    if not token:
        print("Missing GitHub auth token. Set GITHUB_TOKEN or install GitHub App credentials.", file=sys.stderr)
        sys.exit(2)
    repo = get_env("GITHUB_REPOSITORY", required=True)
    event_path = get_env("GITHUB_EVENT_PATH", required=True)

    with open(event_path, "r", encoding="utf-8") as f:
        event = json.load(f)

    pr = event.get("pull_request")
    if not pr:
        print("No pull_request in event payload", file=sys.stderr)
        sys.exit(3)

    pr_number = pr.get("number")
    pr_title = pr.get("title", "")
    pr_body = pr.get("body", "")

    files_url = pr.get("url") + "/files"
    review_mode = get_env("INPUT_REVIEW_MODE", "comment")

    max_files = int(get_env("INPUT_MAX_FILES", "50"))
    max_patch_chars = int(get_env("INPUT_MAX_PATCH_CHARS", "40000"))
    language = get_env("INPUT_LANGUAGE", "de")
    dry_run = get_env("INPUT_DRY_RUN", "false").lower() == "true"
    extra_instructions = get_env("INPUT_EXTRA_INSTRUCTIONS", "")
    comment_title = get_env("INPUT_COMMENT_TITLE", "Hermes PR-Doku & Review")

    files = gh_api_paginate(files_url, token, limit=max_files)

    patch_total = 0
    file_summaries = []
    for f in files:
        patch = f.get("patch") or ""
        if patch_total + len(patch) > max_patch_chars:
            patch = patch[: max(0, max_patch_chars - patch_total)]
        patch_total += len(patch)
        file_summaries.append({
            "filename": f.get("filename"),
            "status": f.get("status"),
            "additions": f.get("additions"),
            "deletions": f.get("deletions"),
            "changes": f.get("changes"),
            "patch": patch,
        })

    prompt = (
        "Du bist Hermes Agent und führst einen strukturierten PR-Review aus.\n"
        "Bitte antworte in diesem Format:\n"
        "---\n"
        "## Kurzfazit\n"
        "<1-3 Sätze>\n\n"
        "## Änderungsübersicht\n"
        "<Was wurde verändert?>\n\n"
        "## Doku-Check\n"
        "- Missing docs: ...\n"
        "- Update needed: ...\n"
        "- No issue: ...\n\n"
        "## Risiken / Bugs / Testlücken\n"
        "- ...\n\n"
        "## Verbesserungsvorschläge\n"
        "1. ...\n"
        "2. ...\n\n"
        "## Nächste PR-Idee\n"
        "<konkrete Idee>\n"
        "---\n\n"
        f"Antwortsprache: {language}\n\n"
        f"PR: {pr_title}\n"
        "Beschreibung:\n"
        f"{pr_body}\n\n"
        "Dateien (gekürzt):\n"
        f"{json.dumps(file_summaries, ensure_ascii=False, indent=2)}\n\n"
        "Zusatz:\n"
        f"{extra_instructions}"
    )

    provider = get_env("INPUT_PROVIDER", required=True)
    model = get_env("INPUT_MODEL", required=True)
    api_key = get_env("INPUT_API_KEY", "")
    base_url = get_env("INPUT_BASE_URL", "")

    env = os.environ.copy()
    if api_key:
        env_key = {
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "google": "GOOGLE_API_KEY",
            "lmstudio": "LM_API_KEY",
        }.get(provider, "PROVIDER_API_KEY")
        env[env_key] = api_key
    if base_url:
        base_env_key = {
            "openrouter": "OPENROUTER_BASE_URL",
            "openai": "OPENAI_BASE_URL",
            "deepseek": "DEEPSEEK_BASE_URL",
            "lmstudio": "LM_BASE_URL",
            "zai": "GLM_BASE_URL",
            "minimax": "MINIMAX_BASE_URL",
            "minimax-cn": "MINIMAX_CN_BASE_URL",
            "alibaba": "DASHSCOPE_BASE_URL",
            "kimi-for-coding": "KIMI_BASE_URL",
            "stepfun": "STEPFUN_BASE_URL",
        }.get(provider, "LM_BASE_URL")
        env[base_env_key] = base_url

    cmd = ["hermes", "chat", "-q", prompt, "-m", model, "--provider", provider, "-Q"]
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        output = res.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        sys.exit(e.returncode)

    comment_body = f"{comment_title}\n\n{output}".strip()

    emit_github_output("comment_body", comment_body)

    if dry_run:
        print(comment_body)
        return

    post_pr_review(token, repo, pr_number, comment_body, review_mode=review_mode)


if __name__ == "__main__":
    main()
