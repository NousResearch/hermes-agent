"""A claimed card's model, resolved once for the dispatcher's pre-spawn refusal and the owner's launch
policy, and the refusal itself. Imports no gateway runner: the dispatcher reads config only."""
import logging
import shlex

logger = logging.getLogger(__name__)


def task_model(task, config) -> tuple[str, bool]:
    """``(model, unresolved)`` the card runs: its ``--model``; the model of the entry its ``--provider``
    names (the pick normalized as a launch reads it, so a bare ``custom`` on a named profile is that
    entry), as the CLI does; else the profile's own (``model.default``, the claimed welcome trial), the
    named endpoint's model, then the provider's catalog default. ``auto`` and a loopback server stay
    ``''`` for the turn to resolve: nothing here probes a credential or a server."""
    from gateway.session_policy import launch_route_pick
    from gateway.session_route_identity import ROUTE_BOUND_KEYS, profile_route_identity
    from hermes_cli.anon_auth import trial_model
    from hermes_cli.models import get_default_model_for_provider
    from hermes_cli.runtime_provider import _loopback_hostname
    from hermes_cli.runtime_provider_custom import _get_named_custom_provider
    from utils import base_url_hostname

    if task.model_override:
        return task.model_override, False
    block = config.get('model')
    held = dict(block) if isinstance(block, dict) else {}
    picked = str(task.provider_override or '').strip()
    pick, keeps_route = launch_route_pick({'provider': picked} if picked else {}, config)
    if 'provider' in pick:
        entry = _get_named_custom_provider(pick['provider'], config=config, resolve_credentials=False) or {}
        if str(entry.get('model') or '').strip():
            return str(entry['model']).strip(), False
        if not keeps_route:
            for stale in ROUTE_BOUND_KEYS:
                held.pop(stale, None)
            held['provider'] = pick['provider']
    if isinstance(block, str) and block.strip():
        return block.strip(), False
    model = held.get('default') or held.get('model')
    model = (model.strip() if isinstance(model, str) else '') or trial_model()
    if model:
        return model, False
    identity = profile_route_identity(dict(config, model=held))
    entry = (_get_named_custom_provider(str(held.get('provider') or ''), config=config, resolve_credentials=False)
             if identity.startswith('custom:') else None) or {}
    model = str(entry.get('model') or '').strip() or ('' if entry or identity == 'auto' else get_default_model_for_provider(identity))
    base_url = str(held.get('base_url') or entry.get('base_url') or '')
    loopback = bool(base_url) and _loopback_hostname(base_url_hostname(base_url))
    return model, not model and identity != 'auto' and not loopback


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
    """The refusal text, its remedy pinned to the assignee's root so any shell edits this profile."""
    from hermes_cli.profiles import profile_root_for_env_home
    from pathlib import Path
    root = Path(home) if profile == 'default' else Path(home).parent.parent
    del profile_root_for_env_home  # same root resolve_profile_env used for ``home``
    return (f"No model configured for profile {profile}; run `HERMES_HOME={shlex.quote(str(root))} hermes -p {profile} model` "
            "or create the task with --model")
