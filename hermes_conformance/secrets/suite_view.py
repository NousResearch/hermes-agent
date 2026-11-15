"""Conformance for a secret store's process-wide view hooks (``SecretStoreProvider.env_view``,
``auth_view``, ``write_view``, ``view_version``, ``bootstrap_env_names``).

These are what ``hermes_secret_routing`` serves the ``.env`` load, the profile secret scope,
``save_env_value`` / ``remove_env_value`` and the ``auth.json`` store from, for a store that keeps
secrets somewhere other than today's files (``files_backed`` false). A files-backed store (the
local backend) never has them called. A backend plugin runs the suite by importing
:class:`SecretViewConformance` into its test module and supplying a ``view_backend`` fixture that
returns a :class:`ViewBackend`.

Covered: a write reads back and moves the view's version; an unset removes the name; an auth
record write reads back and a ``None`` record removes it; a change another writer made reaches the
view after a refresh; a bootstrap name is never written (the write refuses and the view stays
without it); a failed write raises the port's fixed error with no value in its text, its
traceback's frames or any log record, and changes nothing.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from hermes_secret_port import SecretStoreProvider, SecretWriteFailed

MARK = "view-conformance-marker-value-7f3a"
NAME = "VIEW_CONFORMANCE_KEY"
PROVIDER_ID = "view-probe"


@dataclass
class ViewBackend:
    """How the suite drives one store's view hooks.

    - ``provider``: the provider under test, booted for ``home``.
    - ``home``: the profile home the view is for.
    - ``seed_external(env)``: another writer (a person's editor) sets *env* in the store, out of
      band, as the store's own other clients would.
    - ``refresh()``: make the provider pick up changes made elsewhere now (one poll).
    - ``fail_writes()``: while active, the store's writes fail part-way with a plain error.
    """

    provider: SecretStoreProvider
    home: Path
    seed_external: Callable[[dict[str, str]], None]
    refresh: Callable[[], None]
    fail_writes: Callable[[], AbstractContextManager]


def _texts_of(exc: BaseException) -> list[str]:
    texts = [str(exc), repr(exc)]
    tb = exc.__traceback__
    while tb is not None:
        for value in tb.tb_frame.f_locals.values():
            try:
                texts.append(repr(value))
            except Exception:  # health: allow BLE001 -- a local that cannot render holds nothing
                continue
        tb = tb.tb_next
    for chained in (exc.__cause__, exc.__context__):
        if chained is not None:
            texts.extend(_texts_of(chained))
    return texts


class SecretViewConformance:
    """The view-hook invariants; subclasses supply the ``view_backend`` fixture."""

    def test_an_env_write_succeeds_and_reads_back_with_a_new_version(self, view_backend):
        vb = view_backend
        before = vb.provider.view_version(vb.home)
        vb.provider.write_view(vb.home, env_set={NAME: MARK})
        assert vb.provider.env_view(vb.home).get(NAME) == MARK
        assert vb.provider.view_version(vb.home) != before

    def test_an_env_unset_succeeds_and_the_name_is_gone(self, view_backend):
        vb = view_backend
        vb.provider.write_view(vb.home, env_set={NAME: MARK})
        vb.provider.write_view(vb.home, env_unset=(NAME,))
        assert NAME not in vb.provider.env_view(vb.home)

    def test_an_auth_write_succeeds_and_a_none_record_removes_it(self, view_backend):
        vb = view_backend
        vb.provider.write_view(vb.home, auth={PROVIDER_ID: {"access_token": MARK}})
        assert vb.provider.auth_view(vb.home).get(PROVIDER_ID) == {"access_token": MARK}
        vb.provider.write_view(vb.home, auth={PROVIDER_ID: None})
        assert PROVIDER_ID not in vb.provider.auth_view(vb.home)

    def test_a_change_made_elsewhere_reaches_the_view_after_a_refresh(self, view_backend):
        vb = view_backend
        before = vb.provider.view_version(vb.home)
        vb.seed_external({NAME: MARK})
        vb.refresh()
        assert vb.provider.env_view(vb.home).get(NAME) == MARK
        assert vb.provider.view_version(vb.home) != before

    def test_a_bootstrap_name_is_never_written(self, view_backend):
        vb = view_backend
        for name in sorted(vb.provider.bootstrap_env_names()):
            with pytest.raises(Exception):  # any refusal: the point is that nothing lands
                vb.provider.write_view(vb.home, env_set={name: MARK})
            assert name not in vb.provider.env_view(vb.home)

    def test_a_failed_write_is_the_ports_error_and_holds_no_value(self, view_backend, caplog):
        vb = view_backend
        before = dict(vb.provider.env_view(vb.home))
        caplog.set_level(logging.DEBUG)
        with vb.fail_writes(), pytest.raises(SecretWriteFailed) as failed:
            vb.provider.write_view(vb.home, env_set={NAME: MARK})
        assert not [t for t in _texts_of(failed.value) if MARK in t]
        assert MARK not in caplog.text
        assert vb.provider.env_view(vb.home) == before
