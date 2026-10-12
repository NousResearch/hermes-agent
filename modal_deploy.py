import os
import re
import subprocess
try:
    import yaml
except ImportError:
    import hermes_yaml as yaml
import modal

APP_NAME = "hermes-api-server"
HERMES_ROOT = "/workspace/hermes-agent"
HERMES_HOME = "/root/.hermes"

GATEWAY_PORT = 8642

CUSTOM_PROVIDER_ENV = {
    "zenmux": "ZENMUX_API_KEY",
    "opencode": "OPENCODE_API_KEY",
    "nous": "NOUS_API_KEY",
    "cheaperinference": "CHEAPER_INFERENCE_API_KEY",
}
SECRET_NAME_RE = re.compile(
    r"(?:API_KEY|API_TOKEN|TOKEN|SECRET|PASSWORD|PASSPHRASE)$", re.IGNORECASE
)

app = modal.App(APP_NAME)

hermes_volume = modal.Volume.from_name(
    "hermes-storage-v2",
    create_if_missing=True,
)

hermes_secrets = [
    modal.Secret.from_name("hermes-secrets"),
    modal.Secret.from_name("telegram"),
    modal.Secret.from_name("cloudflare"),
    modal.Secret.from_name("github-secret"),
    modal.Secret.from_name("circlecicli"),
    modal.Secret.from_name("hermes-provider-keys"),
    modal.Secret.from_name("modal_proxy_tokens"),
    modal.Secret.from_name("searxng"),
    modal.Secret.from_name("hermes-cloud-mail"),
    modal.Secret.from_name("linear"),
    modal.Secret.from_name("ci-github"),
    modal.Secret.from_name("Instagram"),
]

# صورة Hermes المجهزة بـ Bun و Node.js و gh و wrangler
hermes_image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "ca-certificates",
        "unzip",
        "gnupg",
        "libnss3",
        "libgbm1",
        "libasound2",
        "libx11-xcb1",
        "libxcomposite1",
        "libxdamage1",
        "libxrandr2",
        "libatk1.0-0",
        "libatk-bridge2.0-0",
        "libcups2",
        "libdrm2",
        "libxkbcommon0",
        "libxfixes3",
        "libpango-1.0-0",
        "libcairo2",
    )
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_24.x | bash -",
        "apt-get install -y nodejs",
        "curl -fsSL https://bun.sh/install | bash",
        "ln -s /root/.bun/bin/bun /usr/local/bin/bun",
        "bun upgrade",  # cache-bust: 2026-10-12-force-bun-v1.4.3
        "curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg",
        'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null',
        "apt-get update && apt-get install -y gh",
        "bun add -g wrangler@latest",
        "ln -s /root/.bun/bin/wrangler /usr/local/bin/wrangler",
        'export BUN_INSTALL="$HOME/.bun" && export PATH="$BUN_INSTALL/bin:$PATH" && bun i -g agent-browser@latest',
        "ln -s /root/.bun/bin/agent-browser /usr/local/bin/agent-browser || true",
        "agent-browser install",
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
        f"pip install -e '{HERMES_ROOT}[messaging]' pyyaml",
    )
    .run_commands(
        "pip install --no-cache-dir 'browser-use>=0.13.10,<1'",
        "browser-use install",
    )
)

def build_runtime_environment() -> dict[str, str]:
    """Build the runtime environment without persisting credentials."""
    env = os.environ.copy()
    env["HERMES_HOME"] = HERMES_HOME
    env["TERMINAL_CWD"] = f"{HERMES_HOME}/workspaces"
    env["HERMES_AGENT_TIMEOUT_WARNING"] = "3600"  # 1 hour (3600s)
    env["HERMES_AGENT_TIMEOUT"] = "7200"          # 2 hours (7200s)
    env["GATEWAY_MULTIPLEX_PROFILES"] = "false"

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
    scrub_persisted_secrets()

    config_path = os.path.join(HERMES_HOME, "config.yaml")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        changed = False
        for provider in config.get("custom_providers", []) or []:
            if not isinstance(provider, dict):
                continue
            name = str(provider.get("name", "")).strip().lower()
            env_name = CUSTOM_PROVIDER_ENV.get(name)
            if env_name and provider.get("api_key"):
                provider.pop("api_key", None)
                provider["key_env"] = env_name
                changed = True
        if not isinstance(config.get("lsp"), dict) or config.get("lsp", {}).get("enabled") is not False:
            config.setdefault("lsp", {})["enabled"] = False
            changed = True
        if changed:
            temporary_path = f"{config_path}.tmp.{os.getpid()}"
            with open(temporary_path, "w", encoding="utf-8") as handle:
                os.chmod(temporary_path, 0o600)
                yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, config_path)
            os.chmod(config_path, 0o600)

    env["BROWSER_USE_HEADLESS"] = "true"
    env["AGENT_BROWSER_ARGS"] = "--no-sandbox,--disable-dev-shm-usage"
    env["HERMES_TELEGRAM_DISABLE_FALLBACK_IPS"] = "true"
    env["HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT"] = "3.0"

    return env


def scrub_persisted_secrets() -> None:
    """Remove credentials left by older deployments from the shared volume."""
    paths = [
        os.path.join(HERMES_HOME, ".env"),
        os.path.join(HERMES_HOME, "profiles", "trader", ".env"),
        os.path.join(HERMES_HOME, "profiles", "hazem", ".env"),
        os.path.join(HERMES_HOME, "profiles", "projectsentinelsupport", ".env"),
    ]
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            kept = []
            for line in handle:
                name = line.split("=", 1)[0].strip()
                if name and SECRET_NAME_RE.search(name):
                    continue
                kept.append(line)
        temporary_path = f"{path}.tmp.{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            os.chmod(temporary_path, 0o600)
            handle.writelines(kept)
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)

@app.function(
    image=hermes_image,
    volumes={HERMES_HOME: hermes_volume},
    secrets=hermes_secrets,
    min_containers=1,
    max_containers=1,
    timeout=86400,
    memory=2048,
)
@modal.web_server(
    port=GATEWAY_PORT,
    startup_timeout=300,
)
# عقد تهيئة Browser-Use / Browser:
# أي كود داخل Hermes يقوم بإنشاء BrowserProfile أو Browser يجب أن يمرر:
# - headless=True
# - chromium_sandbox=False
# - args=["--no-sandbox", "--disable-dev-shm-usage"]
def api_server():
    """Run the Hermes messaging gateway and API server."""
    import socket
    import subprocess
    import time
    
    # Reload the volume to get latest config if the container was reused
    hermes_volume.reload()
    
    env = build_runtime_environment()

    env["API_SERVER_ENABLED"] = "true"
    env["API_SERVER_PORT"] = str(GATEWAY_PORT)
    env["API_SERVER_HOST"] = "0.0.0.0"
    env.setdefault("TELEGRAM_ALLOWED_USERS", "*")

    process = subprocess.Popen(
        ["hermes", "gateway", "run"],
        env=env,
        cwd=HERMES_ROOT,
    )

    # Wait until gateway port is listening on localhost before proceeding
    start_time = time.time()
    ready = False
    while time.time() - start_time < 280:
        if process.poll() is not None:
            raise RuntimeError(f"Hermes gateway exited prematurely with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", GATEWAY_PORT), timeout=2.0):
                ready = True
                break
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(1.0)

    if not ready:
        process.terminate()
        raise RuntimeError(f"Hermes gateway failed to bind to 127.0.0.1:{GATEWAY_PORT} within 280s.")

    # Block indefinitely so Modal treats the container as actively executing and never recycles it
    return process.wait()
