"""``onboarding.run``: whether the desktop first-run onboarding opens on the next launch.

The flag lives in the ROOT profile's ``config.yaml`` (``get_default_hermes_root()``), never in the
launch profile's: a pooled backend launched under ``-p work`` must read and write the same file as
one launched under ``default``, because onboarding always targets the default profile.
Unset means "only on a machine that never used Hermes"; ``true`` re-runs it once; ``false`` means done.
"""

from __future__ import annotations

from pathlib import Path

from hermes_constants import get_default_hermes_root


def _root_config() -> Path:
    return get_default_hermes_root() / "config.yaml"


def should_run(*, assume_fresh: bool = False) -> bool:
    """The root config's explicit ``onboarding.run`` (managed overlay included) when it is a bool,
    else whether this machine has never used Hermes (``install_state``; *assume_fresh* is the
    unpackaged desktop's ``--fresh-install`` switch)."""
    value = _explicit_run()
    if value is not None:
        return value
    from hermes_cli.install_state import probe

    return assume_fresh or probe().fresh


def _explicit_run() -> bool | None:
    from hermes_cli.config_effective import load_user_config_effective

    # Not load_config(): it resolves the launch profile and fills the default None for an unset flag.
    section = load_user_config_effective(_root_config()).get("onboarding")
    value = section.get("run") if isinstance(section, dict) else None
    return value if isinstance(value, bool) else None


def latch_first_install() -> bool:
    """The desktop backend's first boot on a fresh install records the decision: ``onboarding.run:
    true`` and the trial reserved. A launch that misses the run (a late answer, the app closed, the
    user chatting first) then stays due until a run ends. True when it wrote them."""
    from hermes_cli.install_state import probe, reserve_trial

    if _explicit_run() is not None or not probe().fresh:
        return False
    set_run(True)  # first: a crash before the reservation still leaves a fresh probe for the claim
    reserve_trial()
    return True


def set_run(value: bool, *, mark_profile_offered: bool = False, step: str | None = None,
            restart: bool = False, sessionless: bool = False) -> None:
    """Persist setup state; sessionless first-start latches preserve another window's progress."""
    if step is not None and step not in ("preferences", "model"):
        raise ValueError("Unknown setup step")
    from agent.onboarding import PROFILE_BUILD_FLAG
    from hermes_cli import config as config_mod

    path = _root_config()
    with config_mod._CONFIG_LOCK:
        config = config_mod.read_user_config_raw(path)
        section = config_mod._ensure_dict(config, "onboarding")
        # Late first-start and Next/Back requests must not reopen a completed guide.
        if value and (sessionless or step is not None) and section.get("run") is False and not restart:
            return
        if sessionless and value and step is None and not restart:
            saved_step = section.get("step")
            step = saved_step if saved_step in ("preferences", "model") else "preferences"
        section["run"] = value
        if step is not None:
            section["step"] = step
        if mark_profile_offered:
            config_mod._ensure_dict(section, "seen")[PROFILE_BUILD_FLAG] = True
        config_mod.atomic_config_write(path, config)
    from hermes_cli import onboarding_due
    onboarding_due.invalidate()


def guide_progress() -> dict:
    from hermes_cli.config_effective import load_user_config_effective

    section = load_user_config_effective(_root_config()).get("onboarding")
    step = section.get("step") if isinstance(section, dict) else None
    return {"step": step if step in ("preferences", "model") else "preferences",
            "started": step in ("preferences", "model")}


def retire_conversational_runs(db) -> list[str]:
    """Retire only setup metadata; never change conversation messages or their cached context."""
    from agent.onboarding_flow import End
    from agent.onboarding_record import KEY, advance, load, open_session_ids

    retired = open_session_ids(db)
    tips = {db.resolve_resume_session_id(sid) or sid for sid in db.session_ids_with_model_config_key(KEY)}
    for session_id in tips:
        record = load(db, session_id)
        if record is None or record.get("sessionless"):
            continue

        advance(db, session_id, End("replaced"), sessionless=True)
    return retired
