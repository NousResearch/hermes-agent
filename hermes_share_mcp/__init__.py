"""Restricted MCP server for a local HermesShare/SMB share root."""

__all__ = ["ShareConfig", "ShareMCPService"]


def __getattr__(name: str):
    if name in __all__:
        from .server import ShareConfig, ShareMCPService

        return {"ShareConfig": ShareConfig, "ShareMCPService": ShareMCPService}[name]
    raise AttributeError(name)
