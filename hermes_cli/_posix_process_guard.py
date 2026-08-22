"""Descendant containment guard for retained POSIX process authority."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Mapping

from hermes_cli import _posix_process_authority_state as S
from hermes_cli import _posix_process_nested as N
from hermes_cli import _posix_process_transfer as T


def _install_detach_helper_adapter() -> None:
    from hermes_cli import _subprocess_compat

    current = _subprocess_compat.windows_detach_popen_kwargs
    if getattr(current, "__hermes_posix_authority_adapter__", False):
        return
    S.original_detach_helper = current

    def authority_detach_popen_kwargs() -> dict[str, object]:
        result = dict(current())
        if not result.get("start_new_session"):
            raise S.ProcessAuthorityError(
                "POSIX detach helper did not request a new session"
            )
        grant = T.begin_process_transfer("hermes-intentional-detached-child")
        result["start_new_session"] = S.TransferStartNewSession(grant)
        return result

    setattr(
        authority_detach_popen_kwargs,
        "__hermes_posix_authority_adapter__",
        True,
    )
    _subprocess_compat.windows_detach_popen_kwargs = authority_detach_popen_kwargs


def install_descendant_guard() -> None:
    if S.guard_installed:
        return

    with S.install_lock:
        if S.guard_installed:
            return

        def guarded_popen_init(self, *args: Any, **kwargs: Any) -> None:
            detach_request = kwargs.get("start_new_session")
            if isinstance(detach_request, S.TransferStartNewSession):
                env = T.desktop_child_env(
                    lifetime=S.LIFETIME_TRANSFERRED,
                    transfer=detach_request.grant,
                    base=kwargs.get("env"),
                )
                kwargs["start_new_session"] = True
                lifetime = S.LIFETIME_TRANSFERRED
            else:
                env, lifetime = T.normalize_child_env(kwargs.get("env"))

            if lifetime != S.LIFETIME_CONTAINED:
                T.launch_transferred_popen(self, args, kwargs, env)
                return

            # A raw private-session request is not an ownership escape. It is
            # a narrower child-control scope that remains owned by the Desktop
            # generation. Keep a retained owner in the caller's group and put
            # only the real target into the private session.
            if kwargs.get("start_new_session"):
                N.launch_nested_owned_popen(self, args, kwargs, env)
                return

            if kwargs.get("process_group") is not None:
                kwargs["process_group"] = None
            preexec_fn = kwargs.get("preexec_fn")
            if preexec_fn in {S.original_setsid, S.original_setpgrp}:
                raise S.ProcessAuthorityError(
                    "contained preexec session creation requires "
                    "start_new_session=True so nested authority can be retained"
                )
            if preexec_fn is not None:
                raise S.ProcessAuthorityError(
                    "contained child preexec_fn is opaque and cannot prove "
                    "process-group containment"
                )

            kwargs["env"] = env
            S.original_popen_init(self, *args, **kwargs)

        subprocess.Popen.__init__ = guarded_popen_init  # type: ignore[assignment]

        if S.original_setsid is not None:

            def guarded_setsid() -> int:
                raise S.ProcessAuthorityError(
                    "direct setsid() escapes retained Desktop authority; "
                    "use a retained Popen start_new_session scope or "
                    "begin_process_transfer()"
                )

            setattr(os, "setsid", guarded_setsid)
        if S.original_setpgid is not None:

            def guarded_setpgid(_pid: int, _pgid: int) -> None:
                raise S.ProcessAuthorityError(
                    "direct setpgid() escapes retained Desktop authority; "
                    "use retained Popen authority"
                )

            setattr(os, "setpgid", guarded_setpgid)
        if S.original_setpgrp is not None:

            def guarded_setpgrp() -> None:
                raise S.ProcessAuthorityError(
                    "direct setpgrp() escapes retained Desktop authority; "
                    "use retained Popen authority"
                )

            setattr(os, "setpgrp", guarded_setpgrp)

        def wrap_posix_spawn(original):
            if original is None:
                return None

            def guarded(path, argv, env, *args: Any, **kwargs: Any):
                child_env, lifetime = T.normalize_child_env(env)
                if lifetime != S.LIFETIME_CONTAINED:
                    token = (child_env.get(S.TRANSFER_TOKEN_ENV) or "").strip()
                    T.revoke_transfer(token)
                    raise S.ProcessAuthorityError(
                        "receipted process transfer requires "
                        "subprocess.Popen acknowledgement"
                    )
                # posix_spawn has no retained owner object to anchor either a
                # transferred or nested private scope. Keep it in the current
                # contained group; callers needing private mutation scope must
                # use Popen so the authority can be retained and receipted.
                if kwargs.get("setsid"):
                    kwargs["setsid"] = False
                if "setpgroup" in kwargs:
                    kwargs.pop("setpgroup")
                return original(path, argv, child_env, *args, **kwargs)

            return guarded

        if S.original_posix_spawn is not None:
            setattr(os, "posix_spawn", wrap_posix_spawn(S.original_posix_spawn))
        if S.original_posix_spawnp is not None:
            setattr(os, "posix_spawnp", wrap_posix_spawn(S.original_posix_spawnp))

        S.guard_installed = True
        _install_detach_helper_adapter()


def install_posix_descendant_guard(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> bool:
    """Install guard-only authority in a contained Python descendant."""

    actual_platform = sys.platform if platform is None else platform
    source = os.environ if environ is None else environ
    marker = (source.get(S.DESCENDANT_GUARD_ENV) or "").strip()
    if not marker:
        return False
    if marker != S.AUTHORITY_MODE:
        raise S.ProcessAuthorityError(
            f"unsupported descendant guard mode: {marker!r}"
        )
    if actual_platform == "win32":
        raise S.ProcessAuthorityError("POSIX descendant guard requested on Windows")
    install_descendant_guard()
    return True


def reset_guard_for_tests() -> None:
    with S.install_lock:
        if not S.guard_installed:
            return
        subprocess.Popen.__init__ = S.original_popen_init  # type: ignore[assignment]
        if S.original_setsid is not None:
            setattr(os, "setsid", S.original_setsid)
        if S.original_setpgid is not None:
            setattr(os, "setpgid", S.original_setpgid)
        if S.original_setpgrp is not None:
            setattr(os, "setpgrp", S.original_setpgrp)
        if S.original_posix_spawn is not None:
            setattr(os, "posix_spawn", S.original_posix_spawn)
        if S.original_posix_spawnp is not None:
            setattr(os, "posix_spawnp", S.original_posix_spawnp)
        if S.original_detach_helper is not None:
            from hermes_cli import _subprocess_compat

            _subprocess_compat.windows_detach_popen_kwargs = S.original_detach_helper
            S.original_detach_helper = None
        S.guard_installed = False
