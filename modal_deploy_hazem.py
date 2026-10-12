import os
import subprocess
import modal
import yaml

APP_NAME = "hermes-hazem-api-server"
HERMES_ROOT = "/workspace/hermes-agent"
HERMES_HOME = "/root/.hermes"

GATEWAY_PORT = 8643

CUSTOM_PROVIDER_ENV = {
    "iamhc": "IAMHC_API_KEY",
    "hcnsec": "HCNSEC_API_KEY",
    "lyclaude": "LYCLAUDE_API_KEY",
    "vyceai": "VYCEAI_API_KEY",
    "nararouter": "NARAROUTER_API_KEY",
    "zenmux": "ZENMUX_API_KEY",
}

app = modal.App(APP_NAME)

hermes_volume = modal.Volume.from_name(
    "hermes-storage",
    create_if_missing=True,
)

hermes_secrets = [
    modal.Secret.from_name("hermes-secrets"),
    modal.Secret.from_name("cloudflare"),
    modal.Secret.from_name("codexeverywhere"),
    modal.Secret.from_name("github-secret"),
    modal.Secret.from_name("OPENROUTER"),
    modal.Secret.from_name("iamhc"),
    modal.Secret.from_name("hermes-provider-keys"),
    modal.Secret.from_name("nabeh-agent"),
    modal.Secret.from_name("modal_proxy_tokens"),
    modal.Secret.from_name("searxng"),
]

hermes_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "ca-certificates",
        "unzip",
        "gnupg",
    )
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_24.x | bash -",
        "apt-get install -y nodejs",
        "curl -fsSL https://bun.sh/install | bash",
        "ln -s /root/.bun/bin/bun /usr/local/bin/bun",
        "curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg",
        'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null',
        "apt-get update && apt-get install -y gh",
        "bun add -g wrangler",
        "ln -s /root/.bun/bin/wrangler /usr/local/bin/wrangler",
    )
    .add_local_dir(
        ".",
        remote_path=HERMES_ROOT,
        copy=True,
        ignore=[
            ".git",
            "node_modules",
            "web/node_modules",
            ".venv",
            "venv",
            "__pycache__",
        ],
    )
    .run_commands(
        f"pip install -e '{HERMES_ROOT}[messaging]'",
    )
)

def build_runtime_environment() -> dict[str, str]:
    """Build the runtime environment without persisting credentials."""
    env = os.environ.copy()
    env["HERMES_HOME"] = HERMES_HOME
    env["TERMINAL_CWD"] = f"{HERMES_HOME}/workspaces"
    env["HERMES_AGENT_TIMEOUT_WARNING"] = "3600"
    env["HERMES_AGENT_TIMEOUT"] = "7200"

    if env.get("GITHUB_TOKEN"):
        token = env["GITHUB_TOKEN"]
        env["GH_TOKEN"] = token
        subprocess.run(
            ["git", "config", "--global", f"url.https://x-access-token:{token}@github.com/.insteadOf", "https://github.com/"],
            env=env,
            check=False,
        )
        subprocess.run(["git", "config", "--global", "user.name", "Hermes Agent"], env=env, check=False)
        subprocess.run(["git", "config", "--global", "user.email", "agent@hermes.dev"], env=env, check=False)

    import secrets
    if not env.get("API_SERVER_KEY") or len(env.get("API_SERVER_KEY", "")) < 16:
        env["API_SERVER_KEY"] = secrets.token_hex(32)

    os.makedirs(HERMES_HOME, exist_ok=True)
    os.makedirs(os.path.join(HERMES_HOME, "workspaces"), exist_ok=True)

    config_path = os.path.join(HERMES_HOME, "config.yaml")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        changed = False
        for provider in config.get("custom_providers", []) or []:
            if not isinstance(provider, dict):
                continue
            env_name = CUSTOM_PROVIDER_ENV.get(str(provider.get("name", "")).strip().lower())
            if env_name and provider.get("api_key"):
                provider.pop("api_key", None)
                provider["key_env"] = env_name
                changed = True
        if changed:
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    env["HERMES_TELEGRAM_DISABLE_FALLBACK_IPS"] = "true"
    env["HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT"] = "3.0"

    return env

@app.function(
    image=hermes_image,
    volumes={HERMES_HOME: hermes_volume},
    secrets=hermes_secrets,
    min_containers=0,
    max_containers=1,
    timeout=86400,
)
@modal.web_server(
    port=GATEWAY_PORT,
    startup_timeout=120,
)
def api_server():
    """Run the Hermes messaging gateway and API server for Hazem Agent."""
    import os
    import subprocess
    
    hermes_volume.reload()
    
    env = build_runtime_environment()

    env["API_SERVER_ENABLED"] = "true"
    env["API_SERVER_PORT"] = str(GATEWAY_PORT)
    env["API_SERVER_HOST"] = "0.0.0.0"
    env["TELEGRAM_ALLOWED_USERS"] = "7839527436"
    env["TELEGRAM_REQUIRE_MENTION"] = "false"

    subprocess.run(
        ["hermes", "-p", "hazem", "gateway", "run"],
        env=env,
        cwd=HERMES_ROOT,
        check=True,
    )
