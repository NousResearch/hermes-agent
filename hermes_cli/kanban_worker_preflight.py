"""A claimed card's model, resolved once for the dispatcher's pre-spawn refusal and the owner's launch
policy, and the refusal itself. Imports no gateway runner: the dispatcher reads config only."""
import logging

logger = logging.getLogger(__name__)


def task_model(task, config) -> tuple[str, bool]:
    """``(model, unresolved)`` the card runs: its ``--model`` (a card's ``--provider`` never comes
    without one); else the profile's own (``model.default``, the claimed welcome trial), the named
    endpoint's model, then the provider's catalog default. ``auto`` and a self-hosted (loopback, LAN)
    server stay ``''`` for the turn to resolve, as the gateway does, before any catalog default, and a
    fallback chain answers an empty primary: nothing here probes a credential or a server."""
    from gateway.session_route_identity import profile_route_identity
    from hermes_cli.anon_auth import trial_model
    from hermes_cli.fallback_config import get_fallback_chain
    from hermes_cli.models import get_default_model_for_provider
    from hermes_cli.runtime_provider_custom import _get_named_custom_provider

    if task.model_override:
        return task.model_override, False
    block = config.get('model')
    if isinstance(block, str) and block.strip():
        return block.strip(), False
    held = dict(block) if isinstance(block, dict) else {}
    model = held.get('default') or held.get('model')
    model = (model.strip() if isinstance(model, str) else '') or trial_model()
    if model:
        return model, False
    identity = profile_route_identity(dict(config, model=held))
    entry = (_get_named_custom_provider(str(held.get('provider') or ''), config=config, resolve_credentials=False)
             if identity.startswith('custom:') else None) or {}
    if str(entry.get('model') or '').strip():
        return str(entry['model']).strip(), False
    if identity == 'auto':
        return '', False
    base_url = _route_base_url(identity, held, entry)
    if base_url and _self_hosted(base_url):
        return '', False
    model = '' if entry else get_default_model_for_provider(identity)
    return model, not model and not get_fallback_chain(config)


def _route_base_url(identity, held, entry) -> str:
    from agent.secret_scope import get_secret_str
    from hermes_cli.local_runtime.endpoint import LLAMACPP_ALIASES

    # A named entry owns its own URL; the bare custom and OpenRouter runtime also honor scoped .env.
    base_url = str(entry.get('base_url') or held.get('base_url') or '').strip()
    if entry:
        return base_url
    if identity not in ('custom', 'openrouter'):
        return _builtin_base_url(identity, base_url, get_secret_str)
    if str(held.get('provider') or '').strip().lower() in LLAMACPP_ALIASES:
        return 'http://127.0.0.1'  # the managed llama.cpp server, whatever model.base_url says (_resolve_llamacpp_runtime)
    return (get_secret_str('CUSTOM_BASE_URL').strip() or base_url
            or get_secret_str('OPENROUTER_BASE_URL').strip())


def _builtin_base_url(identity, base_url, get_secret_str) -> str:
    """Where a built-in provider's turn may go. Its scoped ``*_BASE_URL``, ``model.base_url`` and registry default win
    in a different order on each runtime branch (pool row, alias, Anthropic, Foundry), so a self-hosted one of them
    decides; otherwise the first one set, which is public like the rest."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.runtime_provider_custom import _DIRECT_API_BASE_URLS
    pconfig = PROVIDER_REGISTRY.get(identity)
    env_var = getattr(pconfig, 'base_url_env_var', '') or ('OPENAI_BASE_URL' if identity in _DIRECT_API_BASE_URLS else '')
    urls = [url for url in ((get_secret_str(env_var).strip() if env_var else ''), base_url,
                            str(getattr(pconfig, 'inference_base_url', '') or '')) if url]
    return next((url for url in urls if _self_hosted(url)), urls[0] if urls else '')


def _self_hosted(base_url) -> bool:
    """A loopback, private-network or tailnet (``*.ts.net``) server, or a private-use name: ``*.home.arpa`` plus every
    name model validation already treats as a user's own endpoint (``*.localhost``, ``*.local``, ``*.lan``,
    ``*.internal``, ``*.home``, ``*.localdomain``, single-label). Such a server may pick its own model."""
    import ipaddress

    from agent.model_metadata import is_local_endpoint  # loads the provider registry: only a model-less URL pays
    from hermes_cli.models_validate import _non_public_host
    from utils import base_url_hostname
    host = base_url_hostname(base_url)
    try:
        ipaddress.ip_address(host)
    except ValueError:  # a name: model validation's private-use rules, plus the tailnet and home.arpa suffixes
        return _non_public_host(host) or host.endswith(('.ts.net', '.home.arpa')) or is_local_endpoint(base_url)
    return is_local_endpoint(base_url)  # an IP literal by its scope: a global IPv6 has no dot but is public


class ProfileConfigUnavailable(RuntimeError):
    """The assignee's config backend could not serve it: an infrastructure spawn failure, never "no model"."""


def refuse_unresolved_model(task, workspace: str, board: str | None) -> None:
    """Raise (the caller records ``spawn_failed``) when the owner would refuse the card for a missing
    model, with the reason in the task's worker log. Reads the assignee's effective config only (no
    launch policy, credential or provider probe). A config backend outage (``ConfigBackendUnavailable``,
    a ``SystemExit`` the dispatcher would not catch) raises ``ProfileConfigUnavailable`` instead."""
    from pathlib import Path
    from hermes_cli import kanban_db_dispatch as dispatch
    from hermes_cli.config_backend import ConfigBackendUnavailable
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.kanban_worker_log import open_task_worker_log
    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

    if task.model_override:  # a pinned card runs its own model: no profile read, so no config outage defers it
        return
    profile = normalize_profile_name(task.assignee)
    try:
        home = resolve_profile_env(profile)
    except FileNotFoundError:
        return  # no profile dir: the worker client resolves it (and refuses) itself
    try:
        with dispatch._worker_profile_scope(home):
            _, unresolved = task_model(task, load_user_config_effective(Path(home) / 'config.yaml', fail_closed=True))
    except ConfigBackendUnavailable as exc:
        raise ProfileConfigUnavailable(f"config backend unavailable for profile {profile}: {exc}") from exc
    except Exception:  # health: allow BLE001 -- unreadable config included: the owner decides; logged
        logger.debug("kanban model preflight for %s skipped", task.id, exc_info=True)
        return
    if not unresolved:
        return
    # A refusal spends the retry budget; once it blocks the card, set-model alone does not re-run it.
    unblock = f"hermes kanban{f' --board {board}' if board else ''} unblock {task.id}"  # from any current board
    message = f"{missing_model_detail(profile, home)}; then `{unblock}` if it was blocked"
    with open_task_worker_log(task, board) as log:
        log.write(f"Kanban worker refused: {message}\n".encode())
    raise RuntimeError(message)


def missing_model_detail(profile, home):
    """The refusal text: the card's own remedy first (the owner clips ``detail`` at 160 characters),
    then the profile's, with the assignee's root spelled out rather than in one shell's syntax."""
    from hermes_cli.profiles import profile_root_for_env_home
    from pathlib import Path
    root = profile_root_for_env_home(str(home), Path(home))
    return (f"No model configured for profile {profile}; pin one on this card with `hermes kanban set-model`, "
            f"or run `hermes -p {profile} model` with HERMES_HOME={root}")
