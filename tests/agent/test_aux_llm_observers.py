"""Auxiliary-LLM observer coverage: sync ``call_llm`` and async ``async_call_llm``
both notify registered observers (start/end/error) — the vision path uses the
async entry point, so covering only the sync wrapper would miss vision usage.
Companion to the langfuse plugin's track_aux config (see test_langfuse_plugin.py).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


class TestAuxObserversOnCallLlm:
    def _record_events(self):
        events = []

        def observer(event, kwargs):
            events.append((event, kwargs.get("task"), kwargs.get("model")))

        return events, observer

    def test_sync_call_llm_notifies_start_end(self, monkeypatch):
        from agent import auxiliary_client as aux
        events, observer = self._record_events()
        aux.register_aux_llm_observer(observer)
        try:
            with monkeypatch.context() as m:
                m.setattr(aux, "_call_llm_impl", lambda **kw: SimpleNamespace(model="m"))
                resp = aux.call_llm(task="web_extract", messages=[{"role": "user", "content": "x"}])
        finally:
            aux.unregister_aux_llm_observer(observer)
        assert resp.model == "m"
        assert ("start", "web_extract", None) in events
        assert ("end", "web_extract", None) in events

    def test_sync_call_llm_notifies_error(self, monkeypatch):
        from agent import auxiliary_client as aux
        events, observer = self._record_events()
        aux.register_aux_llm_observer(observer)
        try:
            with monkeypatch.context() as m:
                def boom(**kw):
                    raise RuntimeError("boom")
                m.setattr(aux, "_call_llm_impl", boom)
                with pytest.raises(RuntimeError, match="boom"):
                    aux.call_llm(task="web_extract", messages=[{"role": "user", "content": "x"}])
        finally:
            aux.unregister_aux_llm_observer(observer)
        assert ("start", "web_extract", None) in events
        assert ("error", "web_extract", None) in events

    def test_async_call_llm_notifies_start_end(self, monkeypatch):
        from agent import auxiliary_client as aux
        events, observer = self._record_events()
        aux.register_aux_llm_observer(observer)
        try:
            with monkeypatch.context() as m:
                async def impl(**kw):
                    return SimpleNamespace(model="m")
                m.setattr(aux, "_async_call_llm_impl", impl)
                resp = asyncio.run(
                    aux.async_call_llm(task="vision", messages=[{"role": "user", "content": "x"}])
                )
        finally:
            aux.unregister_aux_llm_observer(observer)
        assert resp.model == "m"
        assert ("start", "vision", None) in events
        assert ("end", "vision", None) in events

    def test_async_call_llm_notifies_error(self, monkeypatch):
        from agent import auxiliary_client as aux
        events, observer = self._record_events()
        aux.register_aux_llm_observer(observer)
        try:
            with monkeypatch.context() as m:
                async def boom(**kw):
                    raise RuntimeError("async boom")
                m.setattr(aux, "_async_call_llm_impl", boom)
                with pytest.raises(RuntimeError, match="async boom"):
                    asyncio.run(
                        aux.async_call_llm(task="vision", messages=[{"role": "user", "content": "x"}])
                    )
        finally:
            aux.unregister_aux_llm_observer(observer)
        assert ("start", "vision", None) in events
        assert ("error", "vision", None) in events

    def test_observer_exception_never_breaks_the_call(self, monkeypatch):
        from agent import auxiliary_client as aux

        def bad_observer(event, kwargs):
            raise RuntimeError("observer bug")

        aux.register_aux_llm_observer(bad_observer)
        try:
            with monkeypatch.context() as m:
                m.setattr(aux, "_call_llm_impl", lambda **kw: SimpleNamespace(model="m"))
                resp = aux.call_llm(task="web_extract", messages=[{"role": "user", "content": "x"}])
            assert resp.model == "m"
        finally:
            aux.unregister_aux_llm_observer(bad_observer)

    def test_no_observers_is_zero_cost_path(self, monkeypatch):
        from agent import auxiliary_client as aux
        assert aux._AUX_LLM_OBSERVERS == []
        with monkeypatch.context() as m:
            m.setattr(aux, "_call_llm_impl", lambda **kw: SimpleNamespace(model="m"))
            resp = aux.call_llm(task="web_extract", messages=[{"role": "user", "content": "x"}])
        assert resp.model == "m"
