#!/usr/bin/env python3
import json
import os
import sys
import subprocess
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


def main():
    token = get_env("GITHUB_TOKEN", required=True)
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
    comments_url = pr.get("comments_url")

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
        "Du bist Hermes Agent. Analysiere den PR und liefere:\n"
        "1) Kurze Zusammenfassung der Änderungen\n"
        "2) Doku-Check: fehlende/zu aktualisierende Doku (README/Docs/Changelog)\n"
        "3) Review-Hinweise (Risiken, Bugs, Testlücken)\n"
        "4) Konkrete Verbesserungsvorschläge für Weiterentwicklung\n"
        "5) Falls passend: Vorschlag für nächste PR-Idee\n\n"
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

    # Set GitHub Action output
    print(f"comment_body<<EOF\n{comment_body}\nEOF")

    if dry_run:
        print(comment_body)
        return

    payload = json.dumps({"body": comment_body}).encode("utf-8")
    req = urllib.request.Request(comments_url, data=payload, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:
        _ = r.read()


if __name__ == "__main__":
    main()
