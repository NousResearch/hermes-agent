"""Shared failure-preserving environment acquisition for live platform policy."""
from pathlib import Path
import re

class PolicyEnvironmentError(RuntimeError):
    """An authoritative environment source could not be acquired."""

def _strict_policy_env(path: Path) -> dict[str, str]:
    """Read dotenv without collapsing absence and acquisition failure."""

    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        raise PolicyEnvironmentError("environment_unavailable") from exc

    from agent.secret_scope import _parse_env_value

    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise PolicyEnvironmentError("environment_unavailable")
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in values:
            raise PolicyEnvironmentError("environment_unavailable")
        value = raw_value.strip()
        if value[:1] in {"'", '"'}:
            quote = value[0]
            escaped = False
            close = None
            for index, char in enumerate(value[1:], 1):
                if quote == '"' and char == "\\" and not escaped:
                    escaped = True
                    continue
                if char == quote and not escaped:
                    close = index
                    break
                escaped = False
            if close is None or (
                value[close + 1 :].strip()
                and not value[close + 1 :].strip().startswith("#")
            ):
                raise PolicyEnvironmentError("environment_unavailable")
            value = value[: close + 1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        try:
            values[key] = _parse_env_value(value)
        except Exception as exc:
            raise PolicyEnvironmentError("environment_unavailable") from exc
    return values

def owner_environment_getter(profile, home, fallback):
    """Managed > pinned-owner dotenv > same-owner active environment.

    Never borrow a routed task's scope/process fallback for a different home.
    The callback preserves the caller's unscoped/multiplex absence contract.
    All acquisition errors propagate to the authorization transaction.
    """
    from hermes_constants import get_hermes_home
    from hermes_cli import managed_scope
    from agent.secret_scope import current_secret_scope, is_multiplex_active

    current_home = get_hermes_home().resolve()
    if home is None:
        if profile is None:
            home = current_home
        else:
            from hermes_cli.profiles import resolve_profile_env
            home = Path(resolve_profile_env(profile))
    owner = Path(home).resolve()
    managed_dir = managed_scope.get_managed_dir()
    managed = _strict_policy_env(managed_dir / ".env") if managed_dir is not None else {}
    values = _strict_policy_env(owner / ".env")
    multiplex = is_multiplex_active()
    scope = current_secret_scope() if owner == current_home else None
    scoped_only = profile is not None or multiplex or owner != current_home
    def get(name):
        if name in managed:
            return managed[name]
        if name in values:
            return values[name]
        # An installed scope is authoritative only for this exact owner home.
        # Preserve explicit None/empty even outside multiplex; never borrow a
        # routed mapping merely because the profile names happen to match.
        if scope is not None and name in scope:
            value = scope.get(name)
            return "" if value is None else str(value)
        return None if scoped_only else fallback(name)
    return get
