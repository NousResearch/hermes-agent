from __future__ import annotations

from gateway.session import PersistedSessionRouteLookup


def run_case(case: dict):
    helper = _route_identity_helper(case["kind"])
    if case["kind"] == "configured":
        if helper is None:
            from gateway.run import GatewayRunner

            result = GatewayRunner._configured_route_identity(case.get("config"))
        else:
            from hermes_cli.providers import infer_api_mode_from_provider

            result = helper(case.get("config"), infer_api_mode_from_provider)
    elif case["kind"] == "persisted":
        store = _store(case["store"])
        if helper is None:
            from gateway.run import GatewayRunner

            runner = object.__new__(GatewayRunner)
            runner.session_store = store
            result = runner._persisted_session_route_identity(case.get("session_key") or "")
        else:
            result = helper(
                store,
                case.get("session_key") or "",
                PersistedSessionRouteLookup,
            )
    else:
        raise AssertionError(f"unknown route_identity case kind: {case['kind']!r}")
    return {"return": _serialize(result), "messages": [], "db": []}


def _route_identity_helper(kind: str):
    try:
        import gateway.fork_ext.route_identity as route_identity
    except ModuleNotFoundError:
        return None
    if kind == "configured":
        return route_identity.configured_route_identity
    if kind == "persisted":
        return route_identity.persisted_session_route_identity
    return None


def _serialize(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def _store(name: str):
    if name == "none":
        return None
    if name == "plain":
        return object()
    if name == "valid":
        return _ValidStore()
    if name == "raises":
        return _RaisingStore()
    if name == "wrong-type":
        return _WrongTypeStore()
    raise AssertionError(f"unknown store fixture: {name!r}")


class _ValidStore:
    def lookup_persisted_route_identity(self, session_key: str):
        return PersistedSessionRouteLookup(
            "valid",
            {"model": f"model:{session_key}", "provider": "openai", "api_mode": "responses"},
        )


class _RaisingStore:
    def lookup_persisted_route_identity(self, session_key: str):
        raise RuntimeError(session_key)


class _WrongTypeStore:
    def lookup_persisted_route_identity(self, session_key: str):
        return {"state": "valid"}
