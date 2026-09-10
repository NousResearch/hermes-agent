"""Count the routes workers actually use before admitting more work."""

from collections import Counter
from pathlib import Path

import yaml

from hermes_cli.kanban_worker_routing import (
    _LOCAL_KANBAN_PROVIDERS,
    _kanban_local_first_enabled,
    _resolve_local_first_route,
)


def _cap(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


class WorkerCapacity:
    def __init__(self, conn, *, model_cap=None, model_caps=None, profile_caps=None):
        from hermes_cli.config import load_config_readonly

        config = (load_config_readonly() or {}).get("kanban") or {}
        self.model_cap = _cap(model_cap if model_cap is not None else config.get("max_in_progress_per_model"))
        self.model_caps = model_caps if model_caps is not None else config.get("max_in_progress_by_model", {})
        self.profile_caps = profile_caps if profile_caps is not None else config.get("max_in_progress_by_profile", {})
        self.local_first = _kanban_local_first_enabled()
        self.routes = {}
        self.models = Counter()
        self.profiles = Counter()
        for row in conn.execute("SELECT * FROM tasks WHERE status = 'running'"):
            self.record(row, row["assignee"])

    def route(self, row, assignee):
        provider, model = row["provider_override"], row["model_override"]
        if provider and model:
            return provider, model
        if assignee not in self.routes:
            from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

            route = None
            try:
                from hermes_cli.config import read_user_config_raw

                path = Path(resolve_profile_env(normalize_profile_name(assignee))) / "config.yaml"
                config = read_user_config_raw(path) or {}
                primary = config.get("model") or {}
                route = (primary.get("provider"), primary.get("default"))
                if self.local_first and not model:
                    route = _resolve_local_first_route(config) or route
            except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):
                pass
            self.routes[assignee] = route
        route = self.routes[assignee]
        if route and all(route):
            return provider or route[0], model or route[1]
        return None

    def limit(self, route):
        limits = [self.model_cap]
        if isinstance(self.model_caps, dict):
            limits.append(_cap(self.model_caps.get("/".join(route))))
        if route[0].lower() in _LOCAL_KANBAN_PROVIDERS:
            limits.append(1)
        return min((value for value in limits if value is not None), default=None)

    def allows(self, row, assignee, result):
        cap = _cap(self.profile_caps.get(assignee)) if isinstance(self.profile_caps, dict) else None
        if cap is not None and self.profiles[assignee] >= cap:
            result.skipped_per_profile_capped.append((row["id"], assignee, self.profiles[assignee]))
            return False
        route = self.route(row, assignee)
        cap = self.limit(route) if route else None
        if cap is not None and self.models[route] >= cap:
            result.skipped_per_model_capped.append((row["id"], *route, self.models[route]))
            return False
        return True

    def record(self, row, assignee):
        self.profiles[assignee] += 1
        route = self.route(row, assignee)
        if route:
            self.models[route] += 1
