"""Source-owned shared custody adapters for cron writers."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import functools
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Iterator

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cron_export_custody import CronCustodyRefused, cron_writer_custody


def _owner_home() -> Any:
    """Resolve the active cron store owner without choosing a caller-supplied path."""
    from cron import jobs

    store = jobs._current_cron_store()
    owner_home = store.cron_dir.parent
    if owner_home.is_symlink():
        raise CronCustodyRefused("cron_custody_profile_symlink")
    if not owner_home.exists():
        jobs.ensure_dirs()
    if not owner_home.is_dir():
        raise CronCustodyRefused("cron_custody_profile_unavailable")
    return owner_home


@contextmanager
def _bound_writer_custody(
    resolve_owner_home: Callable[[], Any], *, ensure_profile: bool = True
) -> Iterator[None]:
    from cron import jobs

    try:
        owner_home = Path(os.path.abspath(os.fspath(resolve_owner_home())))
    except (OSError, TypeError, ValueError) as exc:
        raise CronCustodyRefused("cron_custody_profile_invalid") from exc
    if owner_home.is_symlink():
        raise CronCustodyRefused("cron_custody_profile_symlink")
    if not owner_home.exists():
        if not ensure_profile:
            raise CronCustodyRefused("cron_custody_profile_unavailable")
        try:
            jobs._ensure_cron_dir(owner_home / "cron")
        except OSError as exc:
            raise CronCustodyRefused("cron_custody_profile_unavailable") from exc
    if not owner_home.is_dir():
        raise CronCustodyRefused("cron_custody_profile_unavailable")
    with ExitStack() as stack:
        token = set_hermes_home_override(owner_home)
        try:
            jobs._secure_dir(owner_home)
            stack.enter_context(cron_writer_custody())
        finally:
            reset_hermes_home_override(token)
        yield


def _decorate_writer(
    function: Callable[..., Any],
    resolve_owner_home: Callable[[], Any],
    *,
    ensure_profile: bool,
) -> Callable[..., Any]:
    """Decorate one synchronous source-owned writer."""
    if inspect.iscoroutinefunction(function):
        raise TypeError("cron custody adapters require synchronous writers")
    if inspect.isgeneratorfunction(function):

        @functools.wraps(function)
        def generator_wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
            with _bound_writer_custody(resolve_owner_home, ensure_profile=ensure_profile):
                yield from function(*args, **kwargs)

        return generator_wrapper

    @functools.wraps(function)
    def call_wrapper(*args: Any, **kwargs: Any) -> Any:
        with _bound_writer_custody(resolve_owner_home, ensure_profile=ensure_profile):
            return function(*args, **kwargs)

    return call_wrapper


def writer_custody(function: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a jobs-store writer while spanning its generator lifetime or normal call."""
    return _decorate_writer(function, _owner_home, ensure_profile=True)


def writer_custody_for(
    resolve_owner_home: Callable[[], Any], *, ensure_profile: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Build a synchronous custody decorator for one source-owned profile resolver."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        return _decorate_writer(
            function,
            resolve_owner_home,
            ensure_profile=ensure_profile,
        )

    return decorate


__all__ = ["writer_custody", "writer_custody_for"]
