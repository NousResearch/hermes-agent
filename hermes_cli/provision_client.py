"""One-command client-bot provisioning, in-container, no host access.

A new "sibling" client bot is just a new Hermes profile: the gateway runs a
supervised per-profile s6 service that ``hermes profile create`` registers and
starts live (see hermes_cli/profiles.py). A per-profile gateway runs with
``HERMES_HOME`` pointed at its own profile dir, so it reads the *fixed* var
``TELEGRAM_BOT_TOKEN`` from ``<profile_dir>/.env`` (see config.get_env_value).

This orchestrator wraps the whole flow behind a single deterministic call so a
*cheap* model (e.g. DeepSeek), driven by the ``provision-client`` skill, can run
it reliably:

  1. record the client in the registry (bookkeeping: which clients exist,
     their env + model);
  2. guard: refuse to *activate* a bot with no token (no dead bots), and
     refuse to clone a token-bearing template without an explicit --token
     (the new bot would poll the template's token → Telegram 409, the
     petro-construction failure mode);
  3. create + launch the profile via ``hermes profile create`` (cloning a
     template so the bot inherits a known config -- e.g. a cheap default model);
  4. write the token into the *profile's own* ``.env`` as ``TELEGRAM_BOT_TOKEN``
     -- overriding any value cloned from the template -- then restart that
     gateway so it picks the token up.

Idempotent: re-running for an already-created profile skips creation but still
reconciles the token + restarts, so "stage now, add token later, re-run" works.

Out of scope (host-level, deliberately not automated here): creating sibling
*volumes* / *containers* (client_split, compose_gen). The registry's separate
``/opt/data/secrets/<name>.env`` (``<NAME>_TG_TOKEN``) stub belongs to that
container path, not to the profile path this command provisions.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from hermes_cli.add_client import add_client
from hermes_cli.clients import load_registry, profile_dir

Runner = Callable[[Sequence[str]], None]

# A per-profile gateway reads this fixed env var from its profile's .env.
TELEGRAM_TOKEN_VAR = "TELEGRAM_BOT_TOKEN"

# The claude-proxy reads per-tenant keys from <hermes_root>/proxy/keys.json
# (key -> client_name) and resolves the tenant from the gateway's
# Authorization: Bearer <key>. The gateway sends its provider api_key as that
# bearer; we record it in the profile's .env as the fixed var below so the
# per-profile gateway picks it up the same way it picks up the Telegram token.
PROXY_KEY_VAR = "CLAUDE_PROXY_KEY"
PROXY_KEYS_RELPATH = ("proxy", "keys.json")

# A claude-proxy provider in a profile's config.yaml is recognized by its
# loopback base_url on the proxy port. We wire the tenant key INLINE as
# ``api_key`` — the only spelling the runtime honors across every provider
# config style (bare ``model:`` block, ``providers:``/``custom_providers:``
# entries, and the fallback chain; ``key_env`` is ignored in the bare model
# block — see hermes_cli/runtime_provider.py). The key is loopback-only and
# lives on the same 0600-protected volume as the bot tokens.
PROXY_BASE_URL_MARKER = ":11435"


def _keys_path(hermes_root: str | os.PathLike[str]) -> Path:
    return Path(hermes_root).joinpath(*PROXY_KEYS_RELPATH)


def load_proxy_keys(hermes_root: str | os.PathLike[str]) -> dict[str, str]:
    """Load the key->client map (empty if absent/unreadable)."""
    path = _keys_path(hermes_root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_proxy_keys(hermes_root: str | os.PathLike[str], mapping: dict[str, str]) -> None:
    """Atomically (temp file + os.replace) persist the key map at 0600."""
    path = _keys_path(hermes_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def ensure_proxy_key(
    name: str, hermes_root: str | os.PathLike[str]
) -> tuple[str, bool]:
    """Return ``(key, created)`` for a client. Idempotent: reuse the existing
    key if one already maps to this client, else mint a fresh random key and
    append it to keys.json."""
    mapping = load_proxy_keys(hermes_root)
    for key, client in mapping.items():
        if client == name:
            return key, False
    key = "hk-" + secrets.token_urlsafe(32)
    mapping[key] = name
    write_proxy_keys(hermes_root, mapping)
    return key, True


def _ensure_proxy_api_key(obj: Any, key: str, marker: str) -> int:
    """Recursively set ``api_key`` on every provider-like dict whose
    ``base_url`` targets the proxy and that has no key configured yet. Returns
    the number of entries changed. Never clobbers an explicit ``api_key`` or an
    existing ``key_env``/``api_key_env`` (operator intent wins)."""
    changed = 0
    if isinstance(obj, dict):
        base_url = obj.get("base_url")
        if isinstance(base_url, str) and marker in base_url:
            has_key = bool(str(obj.get("api_key") or "").strip())
            has_keyenv = bool(str(obj.get("api_key_env") or obj.get("key_env") or "").strip())
            if not has_key and not has_keyenv:
                obj["api_key"] = key
                changed += 1
        for value in obj.values():
            changed += _ensure_proxy_api_key(value, key, marker)
    elif isinstance(obj, list):
        for value in obj:
            changed += _ensure_proxy_api_key(value, key, marker)
    return changed


def wire_proxy_provider_key(
    config_path: Path,
    key: str,
    marker: str = PROXY_BASE_URL_MARKER,
) -> bool:
    """Wire the tenant's proxy key into the profile's claude-proxy provider
    block(s) as inline ``api_key`` — the one spelling honored by every runtime
    resolution path — so the gateway sends it with no manual step. Idempotent
    and best-effort: returns True only when the file was actually changed; a
    missing file, no proxy provider block, an already-keyed block, or any
    parse/write error is a no-op (the .env copy of the key is still written, so
    a manual one-line wire remains possible). File mode is preserved restrictive
    (0600) since the config now carries a (loopback-only) secret."""
    if not key:
        return False
    try:
        import yaml  # PyYAML — already a project dependency
    except Exception:
        return False
    try:
        if not config_path.is_file():
            return False
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, (dict, list)):
            return False
        if _ensure_proxy_api_key(cfg, key, marker) == 0:
            return False
        tmp = config_path.with_suffix(config_path.suffix + ".tmp")
        tmp.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.chmod(tmp, 0o600)
        os.replace(tmp, config_path)
        return True
    except Exception:
        return False


def _default_runner(argv: Sequence[str]) -> None:
    subprocess.run(list(argv), check=True)


def token_value(env_path: Path, var: str = TELEGRAM_TOKEN_VAR) -> str | None:
    """Return the non-empty value of ``var`` in an env file, else None."""
    if not env_path.is_file():
        return None
    prefix = f"{var}="
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            return value or None
    return None


def write_token(env_path: Path, value: str, var: str = TELEGRAM_TOKEN_VAR) -> None:
    """Set ``var`` to ``value`` in an env file (0600), preserving other lines."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{var}="
    lines: list[str] = []
    found = False
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(prefix):
                lines.append(f"{var}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{var}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(env_path, 0o600)


def profile_is_created(pdir: Path) -> bool:
    """A profile is 'created' once ``hermes profile create`` has seeded its
    config.yaml (add_client only makes the bare directory)."""
    return (pdir / "config.yaml").is_file()


def build_create_command(name: str, *, clone_from: str | None, description: str | None) -> list[str]:
    cmd = ["hermes", "profile", "create", name]
    if clone_from:
        cmd += ["--clone", "--clone-from", clone_from]
    if description:
        cmd += ["--description", description]
    return cmd


def build_restart_command(name: str) -> list[str]:
    return ["hermes", "gateway", "restart", "--profile", name]


def _clone_source_env_path(
    clone_from: str, hermes_root: str | os.PathLike[str]
) -> Path:
    """``.env`` of a clone-source profile (``default`` is the volume root)."""
    root = Path(hermes_root)
    if clone_from == "default":
        return root / ".env"
    return root / "profiles" / clone_from / ".env"


def provision_client(
    name: str,
    env: str = "prod",
    *,
    token: str | None = None,
    model: str | None = None,
    clone_from: str | None = None,
    description: str | None = None,
    registry_path: str | os.PathLike[str] | None = None,
    hermes_root: str | os.PathLike[str] = "/opt/data",
    require_token: bool = True,
    runner: Runner = _default_runner,
    out=sys.stdout,
) -> bool:
    """Provision (or reconcile) a client bot. Returns True if the profile was
    newly created."""
    # 0. Shared-token footgun guard (the petro-construction failure mode).
    #    `hermes profile create --clone` copies the template's .env verbatim
    #    AND launches the new gateway. If the template embeds a
    #    TELEGRAM_BOT_TOKEN and we are not overriding it with an explicit
    #    --token, the new bot starts polling the *template's* token — a second
    #    poller on one token, which Telegram rejects with 409 "token already in
    #    use" (silently hijacking the template's bot). Each bot needs its own
    #    token, so refuse fast, before we touch the registry. (When --token is
    #    given it overrides the clone, so that path is allowed.)
    if clone_from and not token:
        inherited = token_value(_clone_source_env_path(clone_from, hermes_root))
        if inherited is not None:
            raise ValueError(
                f"refusing to clone '{clone_from}' into '{name}' without "
                f"--token: the template's .env sets {TELEGRAM_TOKEN_VAR}, so "
                f"the new bot would poll the template's token (Telegram 409 "
                f"'token already in use'). Pass a unique --token for '{name}', "
                f"or clone a template that carries no {TELEGRAM_TOKEN_VAR}."
            )

    # 1. Registry bookkeeping. add_client's container-path "set <NAME>_TG_TOKEN"
    #    messaging doesn't apply to the profile flow, so swallow its output and
    #    print profile-correct guidance below.
    add_client(name, env, model=model, registry_path=registry_path,
               hermes_root=hermes_root, out=io.StringIO())
    registry = load_registry(registry_path)
    client = registry.get(name)
    assert client is not None
    print(f"recorded '{name}' ({env}) in registry", file=out)

    pdir = profile_dir(client, hermes_root)        # <hermes_root>/profiles/<name>
    profile_env = pdir / ".env"

    # 1b. Per-tenant proxy key. Idempotent: reuse this client's existing key, or
    #     mint one and record it in <hermes_root>/proxy/keys.json (key ->
    #     client). The claude-proxy resolves the tenant from the Bearer key the
    #     gateway sends as its provider api_key. We also drop the key into the
    #     profile's own .env (the same place the gateway reads TELEGRAM_BOT_TOKEN)
    #     so the gateway can forward it. Secrets never go in the repo / registry.
    proxy_key, key_created = ensure_proxy_key(name, hermes_root)
    if key_created:
        write_token(profile_env, proxy_key, var=PROXY_KEY_VAR)
        print(
            f"  minted proxy key for '{name}' → {_keys_path(hermes_root)} "
            f"(also wrote {PROXY_KEY_VAR} into {profile_env})",
            file=out,
        )
        print(
            f"  (the claude-proxy provider is auto-wired below; if your config "
            f"has no proxy provider block, set its api_key to the value of "
            f"{PROXY_KEY_VAR} so requests are attributed to tenant '{name}')",
            file=out,
        )
    else:
        # Reconcile: make sure the profile .env carries the existing key too.
        if token_value(profile_env, var=PROXY_KEY_VAR) != proxy_key:
            write_token(profile_env, proxy_key, var=PROXY_KEY_VAR)
        print(f"  reusing existing proxy key for '{name}'", file=out)

    # 2. Guard: don't activate a bot with no token (neither passed nor already
    #    present in the profile's .env).
    has_token = bool(token) or token_value(profile_env) is not None
    if require_token and not has_token:
        raise ValueError(
            f"refusing to activate '{name}': no Telegram token. Pass --token, "
            f"put {TELEGRAM_TOKEN_VAR} in {profile_env}, or use "
            "--allow-empty-token to stage without starting the gateway."
        )

    # 3. Create + launch the profile, unless already created.
    newly_created = not profile_is_created(pdir)
    if newly_created:
        cmd = build_create_command(name, clone_from=clone_from, description=description)
        print(f"  creating + launching profile: {' '.join(cmd)}", file=out)
        runner(cmd)
    else:
        print(f"  profile '{name}' already created — reconciling", file=out)

    # 3b. Auto-wire the proxy provider to authenticate as this tenant. config.yaml
    #     exists once the profile is created; set its claude-proxy provider's
    #     inline api_key to this client's key so the gateway sends it with no
    #     manual step. Takes effect on the next gateway (re)start below.
    if wire_proxy_provider_key(pdir / "config.yaml", proxy_key):
        print(
            f"  wired claude-proxy provider api_key for tenant '{name}' "
            f"in {pdir / 'config.yaml'}",
            file=out,
        )

    # 4. Write the token into the profile's own .env (what the gateway reads),
    #    overriding any value cloned from the template, then restart.
    if token:
        write_token(profile_env, token)
        print(f"  wrote {TELEGRAM_TOKEN_VAR} into {profile_env}", file=out)
        runner(build_restart_command(name))
        print(f"provisioned '{name}' ({env}); gateway restarted with token", file=out)
    else:
        print(
            f"staged '{name}' ({env}); add {TELEGRAM_TOKEN_VAR} to {profile_env} "
            f"and run `hermes gateway restart --profile {name}` to activate",
            file=out,
        )
    return newly_created


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hermes-provision-client",
        description="Provision a new client bot (registry + live profile gateway + token).",
    )
    parser.add_argument("name")
    parser.add_argument("--env", default="prod", choices=["prod", "dev"])
    parser.add_argument("--token", help="Telegram bot token value (written to the profile's .env, 0600)")
    parser.add_argument("--model", help="model slug recorded for this client (e.g. a cheap default)")
    parser.add_argument("--clone-from", help="template profile to clone config from")
    parser.add_argument("--description", help="role description for task routing")
    parser.add_argument("--registry", help="path to clients.yaml (default: $HERMES_CLIENTS_REGISTRY)")
    parser.add_argument("--hermes-root", default="/opt/data")
    parser.add_argument("--allow-empty-token", action="store_true",
                        help="stage the client without a token (does not start a working gateway)")
    args = parser.parse_args(argv)

    try:
        provision_client(
            args.name, args.env,
            token=args.token, model=args.model,
            clone_from=args.clone_from, description=args.description,
            registry_path=args.registry, hermes_root=args.hermes_root,
            require_token=not args.allow_empty_token,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
