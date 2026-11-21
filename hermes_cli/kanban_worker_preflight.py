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
    base_url = _route_base_url(identity, held, entry)
    if identity == 'auto' or (base_url and _self_hosted(base_url)):
        return '', False
    model = '' if entry else get_default_model_for_provider(identity)
    return model, not model and not get_fallback_chain(config)


def _route_base_url(identity, held, entry) -> str:
    from agent.secret_scope import get_secret_str

    # A named entry owns its own URL; the bare custom and OpenRouter runtime also honor scoped .env.
    base_url = str(entry.get('base_url') or held.get('base_url') or '').strip()
    if entry or identity not in ('custom', 'openrouter'):
        return base_url
    return (get_secret_str('CUSTOM_BASE_URL').strip() or base_url
            or get_secret_str('OPENROUTER_BASE_URL').strip())


def _self_hosted(base_url) -> bool:
    """A loopback (``localhost.`` too) or private-network server, which may pick its own model."""
    from agent.model_metadata import is_local_endpoint  # loads the provider registry: only a model-less URL pays
    from hermes_cli.runtime_provider import _loopback_hostname
    from utils import base_url_hostname
    return bool(_loopback_hostname(base_url_hostname(base_url))) or is_local_endpoint(base_url)


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
    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

    profile = normalize_profile_name(task.assignee)
    try:
        home = resolve_profile_env(profile)
    except FileNotFoundError:
        return  # no profile dir: the worker client resolves it (and refuses) itself
    try:
        with dispatch._worker_profile_scope(home):
            _, unresolved = task_model(task, load_user_config_effective(Path(home) / 'config.yaml'))
    except ConfigBackendUnavailable as exc:
        raise ProfileConfigUnavailable(f"config backend unavailable for profile {profile}: {exc}") from exc
    except Exception:  # health: allow BLE001 -- the owner builds and refuses this policy itself; logged
        logger.debug("kanban model preflight for %s skipped", task.id, exc_info=True)
        return
    if not unresolved:
        return
    message = missing_model_detail(profile, home)
    with dispatch._open_worker_log(task, board) as log:
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
