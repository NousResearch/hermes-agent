"""A2A sidecar for exposing a narrow, standards-based Hermes agent facade."""

from .config import PeerPolicy, SidecarConfig, load_sidecar_config

__all__ = ["PeerPolicy", "SidecarConfig", "create_app", "load_sidecar_config"]


def create_app(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
    """Lazy import so `hermes-a2a --help` works before the a2a extra is installed."""

    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
